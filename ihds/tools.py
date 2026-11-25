"""The five domain tools, shared by the MCP server and the in-process fallback.

Each function takes JSON-friendly arguments (missing measurements as None) and returns a JSON-friendly
dict, so the same code can be served over MCP (mcp_server/server.py) or called in-process.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import model as risk_model
from .config import FEATURES

TOOL_NAMES = ["risk_score", "explain_contributions", "rag_search", "population_context", "synthetic_neighbors"]


def _frame(record: dict) -> pd.DataFrame:
    row = {}
    for f in FEATURES:
        v = record.get(f)
        row[f] = np.nan if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    return pd.DataFrame([row])[FEATURES]


def make_tool_functions(pipe, kb, welch: pd.DataFrame, synthetic_pool: pd.DataFrame) -> dict:
    imputer, scaler = pipe.named_steps["impute"], pipe.named_steps["scale"]
    pool_z = scaler.transform(synthetic_pool[FEATURES].to_numpy())
    pool_p = pipe.predict_proba(synthetic_pool[FEATURES])[:, 1]

    def risk_score(record: dict) -> dict:
        """Probability of diabetes from the logistic-regression pipeline (Project 3 breast-cancer-classification method).
        `record` maps the 8 Pima feature names to numbers; missing optional values may be null."""
        return {"probability": round(float(pipe.predict_proba(_frame(record))[:, 1][0]), 4)}

    def explain_contributions(record: dict) -> dict:
        """Per-feature log-odds contributions of the logistic-regression model, largest absolute first."""
        clean = {f: float(v) for f, v in _frame(record).iloc[0].items()}
        return {"contributions": [c.__dict__ for c in risk_model.contributions(pipe, clean)]}

    def rag_search(query: str, k: int = 3) -> dict:
        """Retrieve up to k clinical knowledge snippets: ChromaDB embedding search, with the TF-IDF keyword matcher
        (Project 6 agentic-rag-system method) as fallback. `method` says which one answered."""
        evidence, method = kb.retrieve(query, int(k))
        out = {"evidence": evidence, "method": method}
        if method == "chroma":
            out["embedding"] = kb.embedding
        elif kb.method != "keyword" and kb.chroma_error:
            out["fallback_reason"] = kb.chroma_error
        return out

    def population_context(feature: str) -> dict:
        """Welch t-test of one feature, diabetic vs non-diabetic patients (Project 2 statistical-analysis-project method)."""
        row = welch[welch["feature"] == feature].iloc[0]
        return {"feature": feature, "mean_no_diabetes": float(row["mean_no_diabetes"]),
                "mean_diabetes": float(row["mean_diabetes"]), "p_value": float(row["p_value"]),
                "cohens_d": float(row["cohens_d"])}

    def synthetic_neighbors(record: dict, k: int = 25) -> dict:
        """Model probabilities over the k nearest GAN-generated profiles to this patient (Project 5 generative-mnist method)."""
        z = scaler.transform(imputer.transform(_frame(record)))[0]
        idx = np.argsort(((pool_z - z) ** 2).sum(axis=1))[: int(k)]
        p = pool_p[idx]
        return {"k": int(k), "mean": round(float(p.mean()), 3), "min": round(float(p.min()), 3),
                "max": round(float(p.max()), 3), "share_above_0_5": round(float((p >= 0.5).mean()), 3)}

    fns = {f.__name__: f for f in [risk_score, explain_contributions, rag_search, population_context, synthetic_neighbors]}
    assert list(fns) == TOOL_NAMES
    return fns
