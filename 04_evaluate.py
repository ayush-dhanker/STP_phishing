# STAGE 4 (Evaluation) — MODEL SELECTION & CHECKPOINT DECISION
# Modified to:
#   1. Read experiment_results.csv in its new (k-fold) shape:
#        cv_val_f1_mean / test_f1 / test_accuracy / test_roc_auc /
#        holdout_gap_f1 / cv_gap_f1 / overfitting_risk
#      instead of the old single f1_weighted / accuracy / roc_auc columns.
#   2. Select the "best" model by CV VALIDATION score (cv_val_f1_mean),
#      not by test-set score — the old script picked the model that scored
#      highest on the test set, then checked that same test-set score
#      against the pass/fail thresholds below. That's circular: you're
#      using the test set both to choose the winner and to grade it.
#      Selecting on CV instead means the test-set threshold check is a
#      genuinely independent confirmation.
#   3. Prefer candidates NOT flagged overfitting_risk when picking the winner.
#   4. Load train/test from Feast instead of feature_store/*.csv.

import os
import sys
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.base import clone
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score)

try:
    from get_training_data_from_feast import load_split
except ImportError:
    sys.exit("ERROR: get_training_data_from_feast.py not found in this directory.")


# success criteria (from Stage 1 Business Objectives)
THRESH_ACCURACY = 0.95
THRESH_AUC      = 0.95
THRESH_F1       = 0.94

REGISTRY_NAME = "phishing_detector_prod"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

RESULTS_CSV    = "experiment_results.csv"
TRAIN_ENTITIES = "feature_repo/data/train_entities.parquet"
TEST_ENTITIES  = "feature_repo/data/test_entities.parquet"

for path in (RESULTS_CSV, TRAIN_ENTITIES, TEST_ENTITIES):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} missing. Run pipeline_feast.py, 'feast apply', "
                  f"then 03_train_kfold.py first.")


VECTORIZERS = {
    "tfidf_unigram":   TfidfVectorizer(max_features=10_000, ngram_range=(1, 1), stop_words="english"),
    "tfidf_bigram":    TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english"),
    "tfidf_sublinear": TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True),
    "bow_unigram":     CountVectorizer(max_features=10_000, ngram_range=(1, 1), stop_words="english"),
    "bow_bigram":      CountVectorizer(max_features=10_000, ngram_range=(1, 2), stop_words="english"),
    "hashing":         HashingVectorizer(n_features=2**16, ngram_range=(1, 2), stop_words="english", alternate_sign=False),
}
MODELS = {
    "linear_svc":          LinearSVC(max_iter=2000),
    "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1),
    "sgd_log_loss":        SGDClassifier(loss="log_loss", max_iter=200, n_jobs=-1),
    "multinomial_nb":      MultinomialNB(),
    "complement_nb":       ComplementNB(),
    "random_forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1),
}


def main():
    df = pd.read_csv(RESULTS_CSV)

    # --- prefer candidates not flagged as an overfitting risk ---
    safe = df[df["overfitting_risk"] == False]  # noqa: E712 (explicit bool compare is clearer here)
    if len(safe) == 0:
        print("WARNING: every combo was flagged overfitting_risk=True. "
              "Selecting from the full set anyway, but treat results with caution.")
        pool = df
    else:
        if len(safe) < len(df):
            print(f"Excluding {len(df) - len(safe)} overfitting-flagged combo(s) from selection.")
        pool = safe

    pool = pool.sort_values("cv_val_f1_mean", ascending=False)

    # --- TIE-BREAKING RULE (now on CV score, then test AUC) ---
    PROBA_MODELS = {
        "logistic_regression", "sgd_log_loss",
        "multinomial_nb", "complement_nb", "random_forest",
    }

    top_cv_f1 = pool.iloc[0]["cv_val_f1_mean"]
    tied = pool[pool["cv_val_f1_mean"] == top_cv_f1].sort_values("test_roc_auc", ascending=False)

    tied_with_proba = tied[tied["model"].isin(PROBA_MODELS)]
    if len(tied_with_proba) > 0:
        best = tied_with_proba.iloc[0]
    else:
        best = tied.iloc[0]

    vec_name, model_name = best["vectorizer"], best["model"]
    print(f"Tie-break: {len(tied)} configs tied at CV F1={top_cv_f1:.6f}, "
          f"{len(tied_with_proba)} support predict_proba.")

    print("Best model selected from Stage 3 results:")
    print(f"  {vec_name} + {model_name}")
    print(f"  cv_val_f1={best['cv_val_f1_mean']:.4f} (+/- {best['cv_val_f1_std']:.4f})   "
          f"test_accuracy={best['test_accuracy']:.4f}  test_f1={best['test_f1']:.4f}  "
          f"test_auc={best.get('test_roc_auc', float('nan'))}")
    print(f"  overfitting_risk={best['overfitting_risk']}   "
          f"cv_gap_f1={best['cv_gap_f1']:.4f}   holdout_gap_f1={best['holdout_gap_f1']:.4f}")

    # --- checkpoint decision, evaluated on the TRUE holdout test metrics ---
    # (this is now an independent check, since selection above used CV score,
    #  not this same test score)
    auc_val = best.get("test_roc_auc")
    passes = (best["test_accuracy"] >= THRESH_ACCURACY and
              best["test_f1"] >= THRESH_F1 and
              (pd.isna(auc_val) or auc_val >= THRESH_AUC))
    decision = "PROCEED TO DEPLOYMENT" if passes else "RETURN TO BUSINESS UNDERSTANDING"
    print(f"\nCheckpoint decision: {decision}")

    # --- load from Feast (was: pd.read_csv from feature_store/) ---
    train_df = load_split(TRAIN_ENTITIES)
    test_df  = load_split(TEST_ENTITIES)
    Xtr, ytr = train_df["clean_text"].fillna(""), train_df["label"]
    Xte, yte = test_df["clean_text"].fillna(""),  test_df["label"]

    pipeline = Pipeline([
        ("vectorizer", clone(VECTORIZERS[vec_name])),
        ("classifier", clone(MODELS[model_name])),
    ])
    pipeline.fit(Xtr, ytr)
    y_pred = pipeline.predict(Xte)

    with open("best_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("Saved -> best_model.pkl")

    # model registry
    with mlflow.start_run(run_name="stage4_selected_model"):
        mlflow.log_params({"vectorizer": vec_name, "model": model_name})
        mlflow.log_metrics({
            "cv_val_f1_mean": float(best["cv_val_f1_mean"]),
            "test_accuracy": float(best["test_accuracy"]),
            "test_f1": float(best["test_f1"]),
            "holdout_gap_f1": float(best["holdout_gap_f1"]),
        })
        mlflow.set_tag("overfitting_risk", str(best["overfitting_risk"]))
        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=REGISTRY_NAME,
        )

    client = MlflowClient()
    new_version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=REGISTRY_NAME,
        alias="production",
        version=new_version,
    )
    print(f"Registered '{REGISTRY_NAME}' v{new_version} -> alias: production")
    print(f"  Stage 5 can now load it with:")
    print(f'  mlflow.sklearn.load_model("models:/{REGISTRY_NAME}@production")')

    # confusion matrix
    cm = confusion_matrix(yte, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["safe", "phishing"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Best Model: {vec_name} + {model_name}")
    fig.tight_layout()
    fig.savefig("confusion_matrix_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved -> confusion_matrix_best.png")

    # evaluation report
    report = classification_report(yte, y_pred, target_names=["safe", "phishing"], digits=4)
    acc = accuracy_score(yte, y_pred)
    f1w = f1_score(yte, y_pred, average="weighted")

    lines = [
        "# Evaluation Report — Phishing Email Detection\n",
        "## Selected Model",
        f"- Pipeline: **{vec_name} + {model_name}**",
        f"- Selected by: 5-fold CV validation F1 = {best['cv_val_f1_mean']:.4f} "
        f"(+/- {best['cv_val_f1_std']:.4f})",
        f"- Overfitting risk flag: {best['overfitting_risk']} "
        f"(CV train-val gap: {best['cv_gap_f1']:.4f}, train-test gap: {best['holdout_gap_f1']:.4f})\n",
        "## Independent holdout test performance",
        f"- Accuracy: {acc:.4f}",
        f"- F1 (weighted): {f1w:.4f}",
        f"- AUC-ROC (from Stage 3): {auc_val}\n",
        "## Success Criteria (from Stage 1)",
        f"- Accuracy >= {THRESH_ACCURACY}: {'PASS' if acc >= THRESH_ACCURACY else 'FAIL'}",
        f"- F1 >= {THRESH_F1}: {'PASS' if f1w >= THRESH_F1 else 'FAIL'}",
        f"- AUC >= {THRESH_AUC}: "
        f"{'PASS' if (pd.isna(auc_val) or auc_val >= THRESH_AUC) else 'FAIL'}\n",
        f"## Checkpoint Decision\n**{decision}**\n",
        "## Classification Report\n```",
        report, "```\n",
        "## Top 5 candidates considered (by CV validation F1)\n```",
        df.sort_values("cv_val_f1_mean", ascending=False)
            .head(5)[["vectorizer", "model", "cv_val_f1_mean", "test_accuracy",
                      "test_f1", "test_roc_auc", "overfitting_risk"]]
            .to_string(index=False),
        "```",
    ]
    with open("Evaluation_Report.md", "w") as f:
        f.write("\n".join(lines))
    print("Saved -> Evaluation_Report.md")
    print("\nStage 4 complete. Best model ready for Stage 5 deployment.")


if __name__ == "__main__":
    main()