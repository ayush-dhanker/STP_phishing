# STAGE 3 (Modeling / Analysis) — EXPERIMENT TRACKING

import os
import sys
import warnings
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.base import clone
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix)

warnings.filterwarnings("ignore")

FEATURE_STORE = "feature_store"
TRAIN_CSV = f"{FEATURE_STORE}/train_features.csv"
TEST_CSV  = f"{FEATURE_STORE}/test_features.csv"



# Loading From feature store
if not (os.path.exists(TRAIN_CSV) and os.path.exists(TEST_CSV)):
    sys.exit("ERROR: feature store missing. Run 02_data_pipeline.py first.")

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df["clean_text"].fillna("")
y_train = train_df["label"]
X_test  = test_df["clean_text"].fillna("")
y_test  = test_df["label"]

print("Loaded from feature store.")
print("  Train:", X_train.shape[0], "| Test:", X_test.shape[0])


#Vecorizers and Models
vectorizers = {
    "tfidf_unigram":   TfidfVectorizer(max_features=10_000, ngram_range=(1, 1), stop_words="english"),
    "tfidf_bigram":    TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english"),
    "tfidf_sublinear": TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
    "bow_unigram":     CountVectorizer(max_features=10_000, ngram_range=(1, 1), stop_words="english"),
    "bow_bigram":      CountVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english"),
    "hashing":         HashingVectorizer(n_features=2**16, ngram_range=(1, 2), stop_words="english", alternate_sign=False),
}

models = {
    "linear_svc":          LinearSVC(max_iter=2000),
    "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1),
    "sgd_log_loss":        SGDClassifier(loss="log_loss", max_iter=200, n_jobs=-1),
    "multinomial_nb":      MultinomialNB(),
    "complement_nb":       ComplementNB(),
    "random_forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1),
}

SKIP_PAIRS = {
    ("tfidf_sublinear", "multinomial_nb"),
    ("tfidf_sublinear", "complement_nb"),
}


def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":      recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return m


# MLFlow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("phishing_email_classification")


#Experiments 
results = []

for vec_name, vectorizer in vectorizers.items():
    for model_name, model in models.items():
        if (vec_name, model_name) in SKIP_PAIRS:
            print(f"  [SKIP] {vec_name} + {model_name}")
            continue

        run_name = f"{vec_name}__{model_name}"
        print(f"Running: {run_name}")

        with mlflow.start_run(run_name=run_name):
            pipeline = Pipeline([
                ("vectorizer", clone(vectorizer)),
                ("classifier", clone(model)),
            ])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            y_prob = None
            clf = pipeline[-1]
            if hasattr(clf, "predict_proba"):
                try:
                    probs = pipeline.predict_proba(X_test)
                    classes = list(pipeline.classes_)
                    idx = classes.index(1) if 1 in classes else (probs.shape[1] - 1)
                    y_prob = probs[:, idx]
                except Exception:
                    pass
            elif hasattr(clf, "decision_function"):
                y_prob = pipeline.decision_function(X_test)

            metrics = compute_metrics(y_test, y_pred, y_prob)

            mlflow.set_tags({"vectorizer": vec_name, "model": model_name})
            mlflow.log_params({"train_size": len(X_train), "test_size": len(X_test)})
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train[:5], y_pred[:5])
            mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature)

            report = classification_report(
                y_test, y_pred, target_names=["safe_email", "phishing_email"], digits=4)
            mlflow.log_text(report, "classification_report.txt")
            cm = confusion_matrix(y_test, y_pred)
            mlflow.log_text(f"Confusion Matrix:\n{cm}\n(rows=actual, cols=predicted)",
                            "confusion_matrix.txt")

            print(f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}")
            results.append({"run_name": run_name, "vectorizer": vec_name,
                            "model": model_name, **metrics})


# 5. SUMMARY -> experiment_results.csv (will be used in 04_evaluate.py)
results_df = (pd.DataFrame(results)
              .sort_values("f1_weighted", ascending=False)
              .reset_index(drop=True))

print("\n" + "=" * 60)
print("TOP 5 by F1-weighted")
print(results_df.head(5)[["vectorizer", "model", "accuracy", "f1_weighted", "roc_auc"]]
      .to_string(index=False))

results_df.to_csv("experiment_results.csv", index=False)
print("\nSaved -> experiment_results.csv")
print("Stage 3 complete. Next: run 04_evaluate.py")
