"""Explainability (SHAP) and fairness (Fairlearn) for the risk model.

    python -m ihds.mlops.explain

SHAP: `shap.LinearExplainer` on the LogisticRegression step, applied to the imputed + standardized features
(the space the model actually sees), with a background sample of the training split. Values are in log-odds.
Writes the global mean |SHAP| ranking, local explanations for the first few test patients, and a summary
plot (reports/shap_summary.png).

Fairness: the Pima dataset has NO sex, race or ethnicity column (every row is a Pima Indian woman aged 21+),
so no protected attribute other than age can be audited. Age is the attribute used, grouped into the bands
in params.yaml. Fairlearn MetricFrame reports accuracy / recall / precision / selection rate per band on the
test split, plus demographic-parity and equalized-odds differences and ratios. Age is also a model input and
a genuine clinical risk factor, so a higher selection rate in older bands is expected; the numbers are
reported for review, not used to pass or fail the pipeline. Test bands are small (tens of patients).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ..config import FEATURES, SEED, TARGET
from .common import MLOPS_REPORTS, MODEL_PATH, load_params, load_splits, write_json


def _transform(pipe, df: pd.DataFrame) -> np.ndarray:
    return pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(df[FEATURES]))


def shap_explain(pipe, train: pd.DataFrame, test: pd.DataFrame, background: int = 100, local_examples: int = 3,
                 plot: bool = True) -> dict:
    import shap

    bg = _transform(pipe, train.sample(min(background, len(train)), random_state=SEED))
    x_test = _transform(pipe, test)
    explainer = shap.LinearExplainer(pipe.named_steps["clf"], bg)
    sv = explainer(x_test)
    sv.feature_names = FEATURES
    vals = np.asarray(sv.values)
    global_rank = (pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(vals).mean(axis=0)})
                   .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    local = []
    for i in range(min(local_examples, len(test))):
        order = np.argsort(-np.abs(vals[i]))
        local.append({"test_row": i, "base_value_logodds": round(float(np.ravel(sv.base_values)[i]), 4),
                      "sum_shap_logodds": round(float(vals[i].sum()), 4),
                      "prediction_proba": round(float(pipe.predict_proba(test[FEATURES].iloc[[i]])[:, 1][0]), 4),
                      "top_features": [{"feature": FEATURES[j], "shap_logodds": round(float(vals[i, j]), 4),
                                        "value": None if pd.isna(test[FEATURES[j]].iloc[i])
                                        else float(test[FEATURES[j]].iloc[i])} for j in order[:3]]})
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(vals, x_test, feature_names=FEATURES, show=False)
        plt.tight_layout()
        plt.savefig(MLOPS_REPORTS / "shap_summary.png", dpi=110)
        plt.close("all")
    return {"explainer": "LinearExplainer (log-odds, standardized features)", "n_explained": int(len(test)),
            "global": [{"feature": r.feature, "mean_abs_shap": round(float(r.mean_abs_shap), 4)}
                       for r in global_rank.itertuples()],
            "local": local}


def fairness_audit(pipe, test: pd.DataFrame, bins: list, labels: list, threshold: float = 0.5) -> dict:
    from fairlearn.metrics import (
        MetricFrame,
        count,
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
        equalized_odds_ratio,
        false_positive_rate,
        selection_rate,
        true_positive_rate,
    )
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    y = test[TARGET].to_numpy()
    pred = (pipe.predict_proba(test[FEATURES])[:, 1] >= threshold).astype(int)
    age_band = pd.cut(test["Age"], bins=bins, labels=labels, right=False).astype(str)
    mf = MetricFrame(metrics={"count": count, "accuracy": accuracy_score, "recall": recall_score,
                              "precision": lambda a, b: precision_score(a, b, zero_division=0),
                              "selection_rate": selection_rate, "tpr": true_positive_rate,
                              "fpr": false_positive_rate},
                     y_true=y, y_pred=pred, sensitive_features=age_band)
    by_group = mf.by_group.round(4)
    by_group["count"] = by_group["count"].astype(int)
    kw = {"y_true": y, "y_pred": pred, "sensitive_features": age_band}
    return {"sensitive_attribute": "Age band (no sex/race/ethnicity column exists in the data)",
            "bands": labels, "split": "test", "threshold": threshold,
            "overall": {k: round(float(v), 4) for k, v in mf.overall.items()},
            "by_group": by_group.reset_index().rename(columns={"Age": "age_band"}).to_dict(orient="records"),
            "positives_by_group": {str(g): int(y[age_band == g].sum()) for g in labels},
            "demographic_parity_difference": round(float(demographic_parity_difference(**kw)), 4),
            "demographic_parity_ratio": round(float(demographic_parity_ratio(**kw)), 4),
            "equalized_odds_difference": round(float(equalized_odds_difference(**kw)), 4),
            "equalized_odds_ratio": round(float(equalized_odds_ratio(**kw)), 4)}


def run(train: pd.DataFrame | None = None, test: pd.DataFrame | None = None, pipe=None) -> dict:
    p = load_params("explain")
    threshold = load_params("train")["threshold"]
    if train is None or test is None:
        train, test = load_splits()
    pipe = pipe if pipe is not None else joblib.load(MODEL_PATH)
    MLOPS_REPORTS.mkdir(parents=True, exist_ok=True)
    shap_out = shap_explain(pipe, train, test, p["shap_background"], p["local_examples"])
    fair = fairness_audit(pipe, test, p["age_bins"], p["age_labels"], threshold)
    write_json(MLOPS_REPORTS / "shap_global.json", shap_out)
    write_json(MLOPS_REPORTS / "fairness.json", fair)
    print({"shap_top3": [g["feature"] for g in shap_out["global"][:3]],
           "demographic_parity_difference": fair["demographic_parity_difference"],
           "equalized_odds_difference": fair["equalized_odds_difference"]})
    return {"shap": shap_out, "fairness": fair}


if __name__ == "__main__":
    run()
