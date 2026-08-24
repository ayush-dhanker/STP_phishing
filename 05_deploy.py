# STAGE 5 (Deployment) 

import os
import json
import time
import pickle
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import mlflow
import mlflow.sklearn

load_dotenv()

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTRY_NAME = os.getenv("MLFLOW_REGISTRY_MODEL_NAME", "phishing_detector_prod")
FALLBACK_PKL  = "best_model.pkl"

LOG_FILE = os.getenv("MONITOR_LOG_FILE", "monitoring/predictions.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

#Loading the model
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
            "No model found. Run 02 -> 03 -> 04 first to train and register one."
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


@app.post("/predict", response_model=PredictionOutput)
def predict(email: EmailInput):
 
    start_time = time.perf_counter()

    text = email.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Email text is empty.")

   
    label = int(model.predict([text])[0])

    confidence = None
    decision_score = None
    confidence_type = "unavailable"

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))
        confidence_type = "probability"
    elif hasattr(model, "decision_function"):
        decision_score = float(abs(model.decision_function([text])[0]))
        confidence_type = "decision_margin"

    result = {
        "prediction": "phishing email" if label == 1 else "safe email",
        "label": label,
        "confidence": confidence,
        "decision_score": decision_score,
        "confidence_type": confidence_type,
        "model_source": model_source,
    }

    latency_ms = (time.perf_counter() - start_time) * 1000

    log_prediction(text, result, latency_ms)
    return result


# prediction log
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