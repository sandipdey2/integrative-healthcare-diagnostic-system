"""Shared paths and params for the MLOps stages."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import ROOT

PARAMS_PATH = ROOT / "params.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "risk_model.joblib"
MLOPS_REPORTS = ROOT / "reports"
FEATURE_REPO = ROOT / "feature_repo"
FEAST_PARQUET = FEATURE_REPO / "data" / "patient_features.parquet"
MLRUNS_DIR = ROOT / "mlruns"


def load_params(section: str | None = None) -> dict:
    import yaml

    params = yaml.safe_load(PARAMS_PATH.read_text())
    return params[section] if section else params


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (TRAIN_PATH.exists() and TEST_PATH.exists()):
        from .prepare import run as prepare

        prepare()
    return pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
