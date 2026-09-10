# STAGE 3 (Modeling / Analysis) — EXPERIMENT TRACKING

import os
import sys
import warnings
import numpy as np
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix)

warnings.filterwarnings("ignore")

# --- overfitting flag threshold: if (train_score - val_score) exceeds this,
#     for f1_weighted, we flag the combo as "overfitting_risk" ---
OVERFIT_GAP_THRESHOLD = 0.05
N_SPLITS = 5


# ============================================================
# 1. Loading from Feast (was: pd.read_csv from feature_store/)
# ============================================================
try:
    from get_training_data_from_feast import load_split, FEATURES
except ImportError:
    sys.exit("ERROR: get_training_data_from_feast.py not found. "
              "Make sure pipeline_feast.py has been run and 'feast apply' succeeded.")

TRAIN_ENTITIES = "feature_repo/data/train_entities.parquet"
TEST_ENTITIES  = "feature_repo/data/test_entities.parquet"

if not (os.path.exists(TRAIN_ENTITIES) and os.path.exists(TEST_ENTITIES)):
    sys.exit("ERROR: Feast entity files missing. Run pipeline_feast.py first, "
              "then 'feast apply' inside feature_repo/.")

train_df = load_split(TRAIN_ENTITIES)
test_df  = load_split(TEST_ENTITIES)

X_train = train_df["clean_text"].fillna("")
y_train = train_df["label"]
X_test  = test_df["clean_text"].fillna("")
y_test  = test_df["label"]

print("Loaded from Feast feature store.")
print("  Train:", X_train.shape[0], "| Test:", X_test.shape[0])


# ============================================================
# 2. Vectorizers and models (unchanged)
# ============================================================
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


# MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("phishing_email_classification")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
cv_scoring = {
    "accuracy":    "accuracy",
    "f1_weighted": "f1_weighted",
    "roc_auc":     "roc_auc",  # works off predict_proba or decision_function
}


# ============================================================
# 3. Experiments: k-fold CV first, then fit-on-all-train + true holdout
# ============================================================
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

            # --- 3a. K-Fold CV on the training set only ---
            # return_train_score=True gives us, per fold, both the score on
            # the fold's own training portion and on its held-out portion —
            # that gap is a direct, low-variance overfitting signal that
            # doesn't touch the test set at all.
            try:
                cv_results = cross_validate(
                    pipeline, X_train, y_train,
                    cv=cv, scoring=cv_scoring,
                    return_train_score=True,
                    n_jobs=1,  # models already parallelize internally (n_jobs=-1)
                    error_score=np.nan,
                )
                cv_summary = {}
                for metric in cv_scoring:
                    train_scores = cv_results[f"train_{metric}"]
                    val_scores   = cv_results[f"test_{metric}"]
                    cv_summary[f"cv_train_{metric}_mean"] = float(np.nanmean(train_scores))
                    cv_summary[f"cv_val_{metric}_mean"]   = float(np.nanmean(val_scores))
                    cv_summary[f"cv_val_{metric}_std"]    = float(np.nanstd(val_scores))
                    cv_summary[f"cv_gap_{metric}"] = (
                        cv_summary[f"cv_train_{metric}_mean"] - cv_summary[f"cv_val_{metric}_mean"]
                    )
            except Exception as e:
                print(f"  [CV FAILED] {run_name}: {e}")
                cv_summary = {}

            # --- 3b. Fit on the FULL training set, evaluate on train (resubstitution) and true holdout test ---
            pipeline.fit(X_train, y_train)

            y_pred_train = pipeline.predict(X_train)
            y_pred_test  = pipeline.predict(X_test)

            def get_scores(pipe, X, y_true, y_pred):
                y_prob = None
                clf = pipe[-1]
                if hasattr(clf, "predict_proba"):
                    try:
                        probs = pipe.predict_proba(X)
                        classes = list(pipe.classes_)
                        idx = classes.index(1) if 1 in classes else (probs.shape[1] - 1)
                        y_prob = probs[:, idx]
                    except Exception:
                        pass
                elif hasattr(clf, "decision_function"):
                    try:
                        y_prob = pipe.decision_function(X)
                    except Exception:
                        pass
                return compute_metrics(y_true, y_pred, y_prob)

            train_metrics = get_scores(pipeline, X_train, y_train, y_pred_train)
            test_metrics  = get_scores(pipeline, X_test,  y_test,  y_pred_test)

            holdout_gap_f1 = train_metrics["f1_weighted"] - test_metrics["f1_weighted"]
            overfitting_risk = holdout_gap_f1 > OVERFIT_GAP_THRESHOLD or \
                cv_summary.get("cv_gap_f1_weighted", 0) > OVERFIT_GAP_THRESHOLD

            # --- logging ---
            mlflow.set_tags({
                "vectorizer": vec_name,
                "model": model_name,
                "overfitting_risk": str(overfitting_risk),
            })
            mlflow.log_params({
                "train_size": len(X_train),
                "test_size": len(X_test),
                "cv_folds": N_SPLITS,
            })
            mlflow.log_metrics(cv_summary)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric("holdout_gap_f1_weighted", holdout_gap_f1)

            signature = infer_signature(X_train[:5], y_pred_test[:5])
            mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature)

            report = classification_report(
                y_test, y_pred_test, target_names=["safe_email", "phishing_email"], digits=4)
            mlflow.log_text(report, "classification_report.txt")
            cm = confusion_matrix(y_test, y_pred_test)
            mlflow.log_text(f"Confusion Matrix:\n{cm}\n(rows=actual, cols=predicted)",
                             "confusion_matrix.txt")

            flag = "  [OVERFIT RISK]" if overfitting_risk else ""
            print(f"  cv_val_f1={cv_summary.get('cv_val_f1_weighted_mean', float('nan')):.4f}"
                  f"  test_f1={test_metrics['f1_weighted']:.4f}"
                  f"  train-test_gap={holdout_gap_f1:.4f}{flag}")

            results.append({
                "run_name": run_name, "vectorizer": vec_name, "model": model_name,
                "cv_val_f1_mean": cv_summary.get("cv_val_f1_weighted_mean"),
                "cv_val_f1_std":  cv_summary.get("cv_val_f1_weighted_std"),
                "cv_gap_f1":      cv_summary.get("cv_gap_f1_weighted"),
                "train_f1": train_metrics["f1_weighted"],
                "test_f1":  test_metrics["f1_weighted"],
                "test_accuracy": test_metrics["accuracy"],
                "test_roc_auc":  test_metrics.get("roc_auc"),
                "holdout_gap_f1": holdout_gap_f1,
                "overfitting_risk": overfitting_risk,
            })


# ============================================================
# 4. Summary -> experiment_results.csv
# ============================================================
results_df = (pd.DataFrame(results)
              .sort_values("cv_val_f1_mean", ascending=False)
              .reset_index(drop=True))

print("\n" + "=" * 60)
print("TOP 5 by CV validation F1 (most reliable estimate)")
print(results_df.head(5)[["vectorizer", "model", "cv_val_f1_mean", "cv_val_f1_std",
                           "test_f1", "holdout_gap_f1", "overfitting_risk"]]
      .to_string(index=False))

n_flagged = int(results_df["overfitting_risk"].sum())
print(f"\n{n_flagged} / {len(results_df)} combos flagged as possible overfitting "
      f"(train-test F1 gap or CV train-val gap > {OVERFIT_GAP_THRESHOLD})")

results_df.to_csv("experiment_results.csv", index=False)
print("\nSaved -> experiment_results.csv")
print("Stage 3 complete. Next: run 04_evaluate.py")
