"""Descriptive statistics and hypothesis tests.

Adapted from statistical-analysis-project (Heart Failure data): group-wise
descriptive statistics and a Welch two-sample t-test
(`scipy.stats.ttest_ind(..., equal_var=False)`), here applied per feature
between diabetic (Outcome=1) and non-diabetic (Outcome=0) patients.

`welch_tests` also reports Bonferroni and Benjamini-Hochberg (FDR) adjusted
p-values over the eight features. The uncorrected `significant` flag is kept
so existing callers and tests stay unchanged. These tests are run on the
published table (or whichever frame the caller passes); they are descriptive,
not a holdout confirmatory analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import FEATURES, TARGET


def describe_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(TARGET)[FEATURES].agg(["mean", "std"]).round(2)


def adjust_pvalues(p_values, method: str = "bh") -> np.ndarray:
    """Return adjusted p-values. `method` is 'bh' (Benjamini-Hochberg) or 'bonferroni'."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    if method == "bonferroni":
        return np.minimum(1.0, p * m)
    if method != "bh":
        raise ValueError(f"unknown adjustment method: {method}")
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


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
    out = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    out["p_value_bonferroni"] = adjust_pvalues(out["p_value"], "bonferroni")
    out["p_value_bh"] = adjust_pvalues(out["p_value"], "bh")
    out["significant_bonferroni"] = out["p_value_bonferroni"] < alpha
    out["significant_bh"] = out["p_value_bh"] < alpha
    return out
