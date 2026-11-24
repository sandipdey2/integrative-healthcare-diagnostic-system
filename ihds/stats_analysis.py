"""Descriptive statistics and hypothesis tests.

Adapted from statistical-analysis-project (Heart Failure data): group-wise
descriptive statistics and a Welch two-sample t-test
(`scipy.stats.ttest_ind(..., equal_var=False)`), here applied per feature
between diabetic (Outcome=1) and non-diabetic (Outcome=0) patients.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from .config import FEATURES, TARGET


def describe_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(TARGET)[FEATURES].agg(["mean", "std"]).round(2)


def welch_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for col in FEATURES:
        g0 = df.loc[df[TARGET] == 0, col].dropna()
        g1 = df.loc[df[TARGET] == 1, col].dropna()
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        pooled = ((g0.var(ddof=1) + g1.var(ddof=1)) / 2) ** 0.5
        rows.append(
            {
                "feature": col,
                "mean_no_diabetes": round(g0.mean(), 2),
                "mean_diabetes": round(g1.mean(), 2),
                "t_stat": round(float(t), 3),
                "p_value": float(p),
                "cohens_d": round(float((g1.mean() - g0.mean()) / pooled), 3),
                "significant": bool(p < alpha),
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
