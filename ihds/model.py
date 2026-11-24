"""Risk classifier.

Adapted from breast-cancer-classification: stratified split, StandardScaler,
LogisticRegression(max_iter=1000, random_state=42) and the same metric set
(accuracy, precision, recall, F1, confusion matrix). New here: median
imputation for the impossible-zero columns, ROC-AUC, class_weight="balanced"
(to raise recall for the minority diabetic class) and per-feature
contribution scores used by the explanation step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, SEED, TARGET


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")),
        ]
    )


def train(train_df: pd.DataFrame) -> Pipeline:
    pipe = build_pipeline()
    pipe.fit(train_df[FEATURES], train_df[TARGET])
    return pipe


def evaluate(pipe: Pipeline, df: pd.DataFrame, threshold: float = 0.5) -> dict:
    proba = pipe.predict_proba(df[FEATURES])[:, 1]
    pred = (proba >= threshold).astype(int)
    y = df[TARGET].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def predict_proba(pipe: Pipeline, record: dict | pd.DataFrame) -> np.ndarray:
    df = pd.DataFrame([record]) if isinstance(record, dict) else record
    return pipe.predict_proba(df[FEATURES])[:, 1]


def coefficients(pipe: Pipeline) -> pd.DataFrame:
    coef = pipe.named_steps["clf"].coef_[0]
    return (
        pd.DataFrame({"feature": FEATURES, "coefficient": coef, "odds_ratio_per_sd": np.exp(coef)})
        .sort_values("coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class Contribution:
    feature: str
    value: float
    contribution: float  # log-odds contribution relative to the training mean


def contributions(pipe: Pipeline, record: dict) -> list[Contribution]:
    """Per-feature log-odds contribution = coef * standardized value (exact for a linear model)."""
    df = pd.DataFrame([record])[FEATURES]
    imputed = pipe.named_steps["impute"].transform(df)
    z = pipe.named_steps["scale"].transform(imputed)[0]
    coef = pipe.named_steps["clf"].coef_[0]
    items = [Contribution(f, float(imputed[0][i]), float(coef[i] * z[i])) for i, f in enumerate(FEATURES)]
    return sorted(items, key=lambda c: abs(c.contribution), reverse=True)
