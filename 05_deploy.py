# STAGE 5 (Deployment)
# Modified to expose Prometheus metrics at /metrics.
#
# Division of concerns:
#   - Prometheus/Grafana: real-time OPS monitoring — request rate, latency,
#     error rate, prediction counts, confidence distribution. Scraped live
#     from this process.
#   - The JSON prediction log (unchanged) still feeds 06_monitor.py's
#     Jensen-Shannon drift analysis, which needs the raw per-request values
#     replayed against the training reference — not something a handful of
#     Prometheus histogram buckets can reconstruct precisely.

import os
import json
import time
import pickle
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

import mlflow
import mlflow.sklearn

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

load_dotenv()

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTRY_NAME = os.getenv("MLFLOW_REGISTRY_MODEL_NAME", "phishing_detector_prod")
FALLBACK_PKL  = "best_model.pkl"

LOG_FILE = os.getenv("MONITOR_LOG_FILE", "monitoring/predictions.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


# ============================================================
# Prometheus metrics
# ============================================================
PREDICTIONS_TOTAL = Counter(
    "phishing_predictions_total", "Total predictions served", ["prediction"]
)
PREDICTION_ERRORS_TOTAL = Counter(
    "phishing_prediction_errors_total", "Prediction request errors", ["error_type"]
)
REQUEST_LATENCY_SECONDS = Histogram(
    "phishing_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
INPUT_TEXT_LENGTH = Histogram(
    "phishing_input_text_length_chars", "Length of submitted email text (chars)",
    buckets=(50, 100, 200, 500, 1000, 2000, 5000, 10000),
)
PREDICTION_CONFIDENCE = Histogram(
    "phishing_prediction_confidence", "predict_proba confidence when available",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
DECISION_MARGIN = Histogram(
    "phishing_decision_margin", "decision_function margin when available",
    buckets=(0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)


# Loading the model
def load_model():
    """Try the MLflow Model Registry first (the DSPM way).
    If that fails, fall back to the local pickle file."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{REGISTRY_NAME}@production"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from MLflow registry: {model_uri}")
        return model, f"mlflow:{model_uri}"
    except Exception as error:
        print(f"Could not load from registry ({error}).")
        if os.path.exists(FALLBACK_PKL):
            with open(FALLBACK_PKL, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded fallback model: {FALLBACK_PKL}")
            return model, f"pickle:{FALLBACK_PKL}"
        raise RuntimeError(
            "No model found. Run pipeline_feast.py -> 03_train_kfold.py -> "
            "04_evaluate_kfold.py first to train and register one."
        )


model, model_source = load_model()


app = FastAPI(
    title="Phishing Email Detector",
    description="Stage 5 Serving Component of the MLOps-DSPM project.",
    version="1.0",
)


class EmailInput(BaseModel):
    """What the user must send us: just the email text."""
    text: str


class PredictionOutput(BaseModel):
    """What we send back."""
    prediction: str
    label: int
    confidence: float | None
    decision_score: float | None
    confidence_type: str
    model_source: str


@app.get("/")
def health_check():
    """Simple check that the API is alive and which model it uses."""
    return {"status": "ok", "model_source": model_source}


@app.get("/metrics")
def metrics():
    """Prometheus scrapes this endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionOutput)
def predict(email: EmailInput):

    start_time = time.perf_counter()

    text = email.text.strip()
    if not text:
        PREDICTION_ERRORS_TOTAL.labels(error_type="empty_text").inc()
        raise HTTPException(status_code=400, detail="Email text is empty.")

    INPUT_TEXT_LENGTH.observe(len(text))

    label = int(model.predict([text])[0])

    confidence = None
    decision_score = None
    confidence_type = "unavailable"

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        confidence_type = "probability"
        PREDICTION_CONFIDENCE.observe(confidence)
    elif hasattr(model, "decision_function"):
        decision_score = float(abs(model.decision_function([text])[0]))
        confidence_type = "decision_margin"
        DECISION_MARGIN.observe(decision_score)

    result = {
        "prediction": "phishing email" if label == 1 else "safe email",
        "label": label,
        "confidence": confidence,
        "decision_score": decision_score,
        "confidence_type": confidence_type,
        "model_source": model_source,
    }

    latency_ms = (time.perf_counter() - start_time) * 1000

    PREDICTIONS_TOTAL.labels(prediction=result["prediction"]).inc()
    REQUEST_LATENCY_SECONDS.labels(endpoint="predict").observe(latency_ms / 1000)

    log_prediction(text, result, latency_ms)
    return result


# prediction log — unchanged, still feeds 06_monitor.py's drift analysis
def log_prediction(text, result, latency_ms=None):
    """Append one line of JSON per prediction.
    Stage 6 (06_monitor.py) reads this file to watch the live system.
    The email text itself is never stored — only its length."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text_length": len(text),
        "label": result["label"],
        "confidence": result["confidence"],
        "decision_score": result["decision_score"],
        "confidence_type": result["confidence_type"],
        "latency_ms": latency_ms,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)