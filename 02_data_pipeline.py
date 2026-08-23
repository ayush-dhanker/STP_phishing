# STAGE 2 (Data Collection, Exploration, and Preparation) — DATA PIPELINE


import os
import re
import json
import pandas as pd
import numpy as np

DATA_DIR   = os.getenv("DATA_DIR", "data")
TRAIN_FILE = os.path.join(DATA_DIR, "phishing_emails_train.csv")
TEST_FILE  = os.path.join(DATA_DIR, "phishing_emails_test.csv")

FEATURE_STORE = "feature_store"
os.makedirs(FEATURE_STORE, exist_ok=True)


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



# FEATURE ENGINEERING — 
# 20 features
def build_features(df):
    body     = df["body"].fillna("")
    subject  = df["subject"].fillna("")
    sender   = df["sender"].fillna("")
    receiver = df["receiver"].fillna("")

    body_length = body.str.len()
    word_count  = body.str.split().str.len()

    # --- text statistics ---
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

    # --- datetime features (NaN where parsing fails; NO -1 sentinels) ---
    date_col = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["date_parse_failed"] = date_col.isna().astype(int)
    df["send_hour"]  = date_col.dt.hour
    df["send_dow"]   = date_col.dt.dayofweek
    df["is_weekend"] = (df["send_dow"] >= 5).fillna(False).astype(int)
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



# running pipeline
def run_pipeline(input_path):
    df = pd.read_csv(input_path)
    df = clean_dataframe(df)
    df = build_features(df)
    return df


def main():
    print("STAGE 2 — DATA PIPELINE")
    print("=" * 60)

    print(f"Reading raw data from: {DATA_DIR}")
    train = run_pipeline(TRAIN_FILE)
    test  = run_pipeline(TEST_FILE)

    hour_med, dow_med = train["send_hour"].median(), train["send_dow"].median()
    for _df in (train, test):
        _df["send_hour"] = _df["send_hour"].fillna(hour_med).astype(int)
        _df["send_dow"]  = _df["send_dow"].fillna(dow_med).astype(int)

    # columns saving in feature store: text, features, label
    keep = ["clean_text"] + FEATURES + ["label", "email_type"]
    train_out = train[keep].copy()
    test_out  = test[keep].copy()

    train_out.to_csv(f"{FEATURE_STORE}/train_features.csv", index=False)
    test_out.to_csv(f"{FEATURE_STORE}/test_features.csv",  index=False)
    print(f"  Saved -> {FEATURE_STORE}/train_features.csv  {train_out.shape}")
    print(f"  Saved -> {FEATURE_STORE}/test_features.csv   {test_out.shape}")

    write_quality_report(train, test)

    monitor = {
        "train_rows": int(len(train_out)),
        "test_rows":  int(len(test_out)),
        "n_features": len(FEATURES),
        "phishing_ratio_train": float(train_out["label"].mean()),
    }
    with open(f"{FEATURE_STORE}/pipeline_monitor.json", "w") as f:
        json.dump(monitor, f, indent=2)
    print(f"  Saved -> {FEATURE_STORE}/pipeline_monitor.json")
    print("\nStage 2 complete. Next: run 03_train.py")


if __name__ == "__main__":
    main()
