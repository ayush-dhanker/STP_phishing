"""
Replaces: pd.read_csv(feature_store/train_features.csv)

This is what 03_train.py should do now — ask Feast for the point-in-time
correct features joined against your entity+label files.
"""

import pandas as pd
from feast import FeatureStore

from pipeline_feast import FEATURES  # reuse the same feature list

store = FeatureStore(repo_path="feature_repo")

# NOTE: clean_text is included here too — Stage 3's TF-IDF/BoW vectorizers
# run on raw text, not the 20 numeric features, so it has to come along.
feature_refs = [f"email_features:{f}" for f in FEATURES] + ["email_features:clean_text"]


def load_split(entity_parquet_path):
    entity_df = pd.read_parquet(entity_parquet_path)  # email_id, event_timestamp, label, email_type
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()
    return training_df


if __name__ == "__main__":
    train_df = load_split("feature_repo/data/train_entities.parquet")
    test_df  = load_split("feature_repo/data/test_entities.parquet")

    print("Train features:", train_df.shape)
    print("Test features:", test_df.shape)
    print(train_df.head())

    # X/y split, same as before:
    X_train = train_df[FEATURES]
    y_train = train_df["label"]
    X_test  = test_df[FEATURES]
    y_test  = test_df["label"]