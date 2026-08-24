# Development and Operationalization of Data Science Solutions — Detection of Phishing Emails

An end-to-end machine learning system that classifies emails as **phishing**
or **safe**, built following the MLOps-based Data Science Process Model
(MLOps-DSPM) from conceptualization through to monitoring.

Course project · Otto von Guericke University Magdeburg
Research group: Wirtschaftsinformatik / Very Large Business Applications (VLBA)
Advisors: Prof. Dr. Klaus Turowski, M.Sc. Christian Haertel

**Team:** Ayush Dhanker · Suraj Balaji Rautrao · Muhammed Ashiq Nizamudeen · Navyasri Vinjam

---

## What it does

Given the raw text of an email, the system predicts whether it is phishing
or safe and serves that prediction over a REST API. Behind the endpoint sits
a full ML lifecycle: a reproducible data pipeline, tracked experiments, a
versioned model registry, automated deployment tests, and live monitoring
with a continuous-training trigger.

**Model:** TF-IDF (sublinear, 1–2 grams) + LinearSVC
**Performance:** 0.9973 accuracy · 0.9973 F1 · 0.99999 AUC on the held-out
test set.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

python run_pipeline.py              # stages 2-4  (~10 min)
uvicorn 05_deploy:app --port 8000   # stage 5: serving
```

Interactive API docs: <http://localhost:8000/docs>

```bash
# stage 6, with the API running in another terminal
python send_test_predictions.py     # simulate live requests
python 06_monitor.py                # check for drift
python ct_retrain.py --dry-run      # preview the retrain trigger
```

Browse experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

---

## Pipeline

Scripts are numbered by DSPM stage and run in sequence — each reads the
previous stage's output.

| Script | Stage | Role |
|---|---|---|
| `01_eda.py` | 2a | Exploratory data analysis |
| `02_data_pipeline.py` | 2b | Parse, clean, feature engineering → feature store |
| `03_train.py` | 3 | 34 experiments, tracked in MLflow |
| `04_evaluate.py` | 4 | Select best model, checkpoint decision, register |
| `05_deploy.py` | 5 | FastAPI serving component |
| `06_monitor.py` | 6 | Drift monitoring |
| `ct_retrain.py` | 6 | Continuous-training trigger |

`run_pipeline.py` runs stages 2–4 in order. Data is a **frozen local
snapshot** of the HuggingFace dataset `drorrabin/phishing_emails-data`
(26,946 train / 3,705 test), committed so every run uses identical data.

---

## Tech stack

Python · scikit-learn · MLflow · FastAPI · Docker · pytest


---

## License

MIT