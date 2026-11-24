"""Data loading, cleaning and splitting.

Cleaning style (explicit, reusable functions + missing-value audit) follows the
data-preparation approach of the statistical-analysis-project and
breast-cancer-classification notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, FEATURES, SEED, TARGET, TEST_SIZE, ZERO_AS_MISSING


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    return df[FEATURES + [TARGET]].copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN (imputed later inside the model pipeline)."""
    out = df.copy()
    for col in ZERO_AS_MISSING:
        out[col] = out[col].replace(0, np.nan)
    return out


def missing_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Count of impossible zeros per column in the raw data."""
    rows = []
    for col in FEATURES:
        zeros = int((df[col] == 0).sum()) if col in ZERO_AS_MISSING else 0
        rows.append({"feature": col, "zero_as_missing": zeros, "share": round(zeros / len(df), 3)})
    return pd.DataFrame(rows)


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Stratified train/test split (same pattern as breast-cancer-classification)."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return train.reset_index(drop=True), test.reset_index(drop=True)
