"""DVC stage `prepare`: clean the raw CSV, write the stratified train/test split and the Feast source.

    python -m ihds.mlops.prepare

Reads data/pima_diabetes.csv (kept in git because the Streamlit / Azure app reads it directly) and writes
data/processed/{train,test}.parquet and feature_repo/data/patient_features.parquet (all DVC outputs, not in git).
patient_id is the row number in the CSV; the dataset has no real patient identifiers or timestamps.
"""
from __future__ import annotations

import pandas as pd

from .. import data
from ..config import FEATURES, TARGET
from .common import FEAST_PARQUET, PROCESSED_DIR, TEST_PATH, TRAIN_PATH, load_params


def feast_frame(df: pd.DataFrame, event_timestamp: str) -> pd.DataFrame:
    out = df[FEATURES + [TARGET]].copy()
    out.insert(0, "patient_id", range(len(out)))
    out["patient_id"] = out["patient_id"].astype("int64")
    for c in FEATURES:
        out[c] = out[c].astype("float64")
    out[TARGET] = out[TARGET].astype("int64")
    out["event_timestamp"] = pd.Timestamp(event_timestamp)
    return out


def run() -> dict:
    p = load_params("prepare")
    df = data.load_clean()
    train, test = data.split(df, test_size=p["test_size"], seed=p["seed"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    FEAST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    feast_frame(df, p["feast_event_timestamp"]).to_parquet(FEAST_PARQUET, index=False)
    out = {"rows": len(df), "train_rows": len(train), "test_rows": len(test),
           "train_positive_rate": round(float(train[TARGET].mean()), 4)}
    print(out)
    return out


if __name__ == "__main__":
    run()
