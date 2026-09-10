# STAGE 2 (Data Collection, Exploration, and Preparation) — DATA PIPELINE
# Modified to feed a Feast feature store instead of local CSV files.
#
# Key differences from the original script:
#   1. Every row gets an `email_id` (entity key) and `event_timestamp` — Feast
#      requires both on every feature source.
#   2. Train and test are concatenated into ONE feature table
#      (feature_repo/data/email_features.parquet). Feast stores feature
#      values keyed by entity+time; it doesn't have a notion of "train file"
#      vs "test file".
#   3. The train/test split lives in two small "entity + label" files
#      (train_entities.parquet, test_entities.parquet) containing just
#      email_id, event_timestamp, label, email_type. These are what you hand
#      to Feast's get_historical_features() to pull back a point-in-time
#      correct training/test set.
#   4. Everything is written as parquet, not CSV — Feast's FileSource needs a
#      typed timestamp column, which CSV round-trips badly.

import os
import re
import json
import pandas as pd
import numpy as np

DATA_DIR   = os.getenv("DATA_DIR", "data")
TRAIN_FILE = os.path.join(DATA_DIR, "phishing_emails_train.csv")
TEST_FILE  = os.path.join(DATA_DIR, "phishing_emails_test.csv")

FEATURE_REPO = "feature_repo"
FEATURE_DATA_DIR = os.path.join(FEATURE_REPO, "data")
os.makedirs(FEATURE_DATA_DIR, exist_ok=True)


# split the column into 5 fields
def parse_email(raw_text):
    def extract(pattern, text, default=""):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else default

    date     = extract(r"Date:\s*(.+?)(?:\n|Sender:)",            raw_text)
    sender   = extract(r"Sender:\s*(.+?)(?:\n|Receiver:)",         raw_text)
    receiver = extract(r"Receiver:\s*(.+?)(?:\n|Email Subject:)",  raw_text)
    subject  = extract(r"Email Subject:\s*(.+?)(?:\n|Email Body:)", raw_text)
    body     = extract(r"Email Body:\s*(.+?)(?:\nEmail type is:|$)", raw_text)
    return date, sender, receiver, subject, body


def clean_dataframe(df):
    parsed = df["text"].apply(
        lambda raw: pd.Series(
            parse_email(str(raw)),
            index=["date", "sender", "receiver", "subject", "body"],
        )
    )
    df = pd.concat([df.reset_index(drop=True), parsed], axis=1)

    df["body"]    = df["body"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["subject"] = df["subject"].str.replace(r"\s+", " ", regex=True).str.strip()

    df["clean_text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

    before = len(df)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed} rows with empty clean_text (parsing failed)")

    df["label"] = (df["email_type"] == "phishing email").astype(int)
    return df


# FEATURE ENGINEERING — 20 features
def build_features(df):
    body     = df["body"].fillna("")
    subject  = df["subject"].fillna("")
    sender   = df["sender"].fillna("")
    receiver = df["receiver"].fillna("")

    body_length = body.str.len()
    word_count  = body.str.split().str.len()

    df["body_len"]        = body_length.fillna(0)
    df["subject_len"]     = subject.str.len().fillna(0)
    df["word_count"]      = word_count.fillna(0)
    df["char_per_word"]   = body_length / (word_count + 1)
    df["digit_ratio"]     = body.str.count(r"\d") / (body_length + 1)
    df["special_ratio"]   = body.str.count(r"[!$%&*@#]") / (body_length + 1)
    df["uppercase_ratio"] = body.apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )

    df["num_url"]       = body.str.count(r"https?://")
    df["has_url"]       = (df["num_url"] > 0).astype(int)
    df["is_free_email"] = sender.str.contains(
        r"gmail|yahoo|hotmail|outlook", case=False, na=False).astype(int)
    df["subj_exclamation"] = subject.str.contains("!", na=False).astype(int)
    df["subj_question"]    = subject.str.contains(r"\?", na=False, regex=True).astype(int)
    df["subj_urgent"] = subject.str.contains(
        r"urgent|action required|verify|confirm|suspended|limited",
        case=False, na=False, regex=True).astype(int)
    df["subj_money"] = subject.str.contains(
        r"free|win|prize|cash|offer|deal|discount",
        case=False, na=False, regex=True).astype(int)
    df["body_urgent"] = body.str.contains(
        r"click here|verify now|login|password|account.{0,20}suspend",
        case=False, na=False, regex=True).astype(int)
    df["num_recipients"] = receiver.str.count(",") + 1

    # --- datetime features ---
    date_col = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date_parse_failed"] = date_col.isna().astype(int)
    df["send_hour"]  = date_col.dt.hour
    df["send_dow"]   = date_col.dt.dayofweek
    df["is_weekend"] = (df["send_dow"] >= 5).fillna(False).astype(int)

    # --- Feast requirements: entity key + event timestamp ---
    # Keep the parsed send time as the event_timestamp where we have it;
    # fall back to "now" (ingestion time) where parsing failed, so every
    # row still has a valid, typed timestamp for the offline store.
    df["event_timestamp"] = date_col.fillna(pd.Timestamp.now(tz="UTC"))

    return df


FEATURES = [
    "body_len", "subject_len", "word_count", "char_per_word",
    "digit_ratio", "special_ratio", "uppercase_ratio",
    "num_url", "has_url", "is_free_email",
    "subj_exclamation", "subj_question", "subj_urgent", "subj_money",
    "body_urgent", "num_recipients",
    "send_hour", "send_dow", "is_weekend", "date_parse_failed",
]


# data quality report
def write_quality_report(train, test):
    lines = ["# Data Quality Report — Phishing Email Detection\n"]
    for name, df_ in [("Train", train), ("Test", test)]:
        lines.append(f"## {name} set\n")
        lines.append(f"- Rows: {len(df_):,}")
        counts = df_["email_type"].value_counts()
        lines.append("- Class distribution:")
        for k, v in counts.items():
            lines.append(f"    - {k}: {v:,} ({v/len(df_)*100:.1f}%)")
        lines.append(f"- Duplicate raw texts: {df_['text'].duplicated().sum():,}")
        empty = (df_["clean_text"].fillna("").str.strip() == "").sum()
        lines.append(f"- Empty clean_text rows: {empty:,}")
        lines.append(f"- Date parse failures: {int(df_['date_parse_failed'].sum()):,}\n")
    with open("Data_Quality_Report.md", "w") as f:
        f.write("\n".join(lines))
    print("  Saved -> Data_Quality_Report.md")


def run_pipeline(input_path):
    df = pd.read_csv(input_path)
    df = clean_dataframe(df)
    df = build_features(df)
    return df


def main():
    print("STAGE 2 — DATA PIPELINE (Feast-backed)")
    print("=" * 60)

    print(f"Reading raw data from: {DATA_DIR}")
    train = run_pipeline(TRAIN_FILE)
    test  = run_pipeline(TEST_FILE)

    hour_med, dow_med = train["send_hour"].median(), train["send_dow"].median()
    for _df in (train, test):
        _df["send_hour"] = _df["send_hour"].fillna(hour_med).astype(int)
        _df["send_dow"]  = _df["send_dow"].fillna(dow_med).astype(int)

    # --- assign globally-unique entity IDs across train + test ---
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    train["email_id"] = [f"train_{i}" for i in range(len(train))]
    test["email_id"]  = [f"test_{i}"  for i in range(len(test))]

    # --- 1) the feature table Feast serves from (all rows, one source) ---
    feature_cols = ["email_id", "event_timestamp", "clean_text"] + FEATURES
    all_features = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
    all_features.to_parquet(
        os.path.join(FEATURE_DATA_DIR, "email_features.parquet"), index=False
    )
    print(f"  Saved -> {FEATURE_DATA_DIR}/email_features.parquet  {all_features.shape}")

    # --- 2) small entity+label files, one per split, used to pull the ---
    #        point-in-time correct train/test sets back out of Feast
    entity_cols = ["email_id", "event_timestamp", "label", "email_type"]
    train[entity_cols].to_parquet(
        os.path.join(FEATURE_DATA_DIR, "train_entities.parquet"), index=False
    )
    test[entity_cols].to_parquet(
        os.path.join(FEATURE_DATA_DIR, "test_entities.parquet"), index=False
    )
    print(f"  Saved -> {FEATURE_DATA_DIR}/train_entities.parquet  {train[entity_cols].shape}")
    print(f"  Saved -> {FEATURE_DATA_DIR}/test_entities.parquet   {test[entity_cols].shape}")

    write_quality_report(train, test)

    monitor = {
        "train_rows": int(len(train)),
        "test_rows":  int(len(test)),
        "n_features": len(FEATURES),
        "phishing_ratio_train": float(train["label"].mean()),
    }
    with open(os.path.join(FEATURE_DATA_DIR, "pipeline_monitor.json"), "w") as f:
        json.dump(monitor, f, indent=2)
    print(f"  Saved -> {FEATURE_DATA_DIR}/pipeline_monitor.json")

    print("\nStage 2 complete. Next:")
    print("  cd feature_repo && feast apply")
    print("  feast materialize-incremental $(date +%Y-%m-%d)")
    print("  then run 03_train.py, which pulls features via get_historical_features()")


if __name__ == "__main__":
    main()
