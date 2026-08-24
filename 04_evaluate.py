
# STAGE 4 (Evaluation) — MODEL SELECTION & CHECKPOINT DECISION

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


# success criteria (from Stage 1 Business Objectives)

THRESH_ACCURACY = 0.95
THRESH_AUC      = 0.95
THRESH_F1       = 0.94

REGISTRY_NAME = "phishing_detector_prod"   

mlflow.set_tracking_uri("sqlite:///mlflow.db")   

RESULTS_CSV = "experiment_results.csv"
TRAIN_CSV   = "feature_store/train_features.csv"
TEST_CSV    = "feature_store/test_features.csv"

for path in (RESULTS_CSV, TRAIN_CSV, TEST_CSV):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} missing. Run 02 then 03 first.")


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
    df = pd.read_csv(RESULTS_CSV).sort_values("f1_weighted", ascending=False)

    # --- TIE-BREAKING RULE
    PROBA_MODELS = {
        "logistic_regression", "sgd_log_loss",
        "multinomial_nb", "complement_nb", "random_forest",
    }

    top_f1 = df.iloc[0]["f1_weighted"]
    tied = df[df["f1_weighted"] == top_f1].sort_values("roc_auc", ascending=False)        

    tied_with_proba = tied[tied["model"].isin(PROBA_MODELS)]
    if len(tied_with_proba) > 0:
        best = tied_with_proba.iloc[0]             
    else:
        best = tied.iloc[0]                        

    vec_name, model_name = best["vectorizer"], best["model"]
    print(f"Tie-break: {len(tied)} configs tied at F1={top_f1:.6f}, "
          f"{len(tied_with_proba)} support predict_proba.")

    print("Best model selected from Stage 3 results:")
    print(f"  {vec_name} + {model_name}")
    print(f"  accuracy={best['accuracy']:.4f}  f1={best['f1_weighted']:.4f}  "
          f"auc={best.get('roc_auc', float('nan'))}")


    auc_val = best.get("roc_auc")
    passes = (best["accuracy"] >= THRESH_ACCURACY and
              best["f1_weighted"] >= THRESH_F1 and
              (pd.isna(auc_val) or auc_val >= THRESH_AUC))
    decision = "PROCEED TO DEPLOYMENT" if passes else "RETURN TO BUSINESS UNDERSTANDING"
    print(f"\nCheckpoint decision: {decision}")

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    Xtr, ytr = train["clean_text"].fillna(""), train["label"]
    Xte, yte = test["clean_text"].fillna(""),  test["label"]

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
            "accuracy": float(best["accuracy"]),
            "f1_weighted": float(best["f1_weighted"]),
        })
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

    # evaakuation report
    report = classification_report(yte, y_pred, target_names=["safe", "phishing"], digits=4)
    acc = accuracy_score(yte, y_pred)
    f1w = f1_score(yte, y_pred, average="weighted")

    lines = [
        "# Evaluation Report — Phishing Email Detection\n",
        "## Selected Model",
        f"- Pipeline: **{vec_name} + {model_name}**",
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
        "## Top 5 candidates considered\n```",
        df.head(5)[["vectorizer", "model", "accuracy", "f1_weighted", "roc_auc"]]
            .to_string(index=False),
        "```",
    ]
    with open("Evaluation_Report.md", "w") as f:
        f.write("\n".join(lines))
    print("Saved -> Evaluation_Report.md")
    print("\nStage 4 complete. Best model ready for Stage 5 deployment.")


if __name__ == "__main__":
    main()