"""
Feast feature definitions for the phishing-email detector.

Run from inside feature_repo/:
    feast apply
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# --- Entity: what a "row" of features is about ---
email = Entity(
    name="email_id",
    join_keys=["email_id"],
    description="Unique identifier for a single email (train_<i> / test_<i>)",
)

# --- Source: where the feature values live (parquet, produced by pipeline_feast.py) ---
email_features_source = FileSource(
    name="email_features_source",
    path="data/email_features.parquet",
    timestamp_field="event_timestamp",
)

# --- Feature view: the 20 engineered features, keyed by email_id + timestamp ---
email_features_view = FeatureView(
    name="email_features",
    entities=[email],
    ttl=timedelta(days=3650),  # effectively "no expiry" for this offline dataset
    schema=[
        Field(name="body_len", dtype=Float32),
        Field(name="subject_len", dtype=Float32),
        Field(name="word_count", dtype=Float32),
        Field(name="char_per_word", dtype=Float32),
        Field(name="digit_ratio", dtype=Float32),
        Field(name="special_ratio", dtype=Float32),
        Field(name="uppercase_ratio", dtype=Float32),
        Field(name="num_url", dtype=Int64),
        Field(name="has_url", dtype=Int64),
        Field(name="is_free_email", dtype=Int64),
        Field(name="subj_exclamation", dtype=Int64),
        Field(name="subj_question", dtype=Int64),
        Field(name="subj_urgent", dtype=Int64),
        Field(name="subj_money", dtype=Int64),
        Field(name="body_urgent", dtype=Int64),
        Field(name="num_recipients", dtype=Int64),
        Field(name="send_hour", dtype=Int64),
        Field(name="send_dow", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="date_parse_failed", dtype=Int64),
        Field(name="clean_text", dtype=String),  # kept for reference / NLP baselines
    ],
    online=True,
    source=email_features_source,
)
