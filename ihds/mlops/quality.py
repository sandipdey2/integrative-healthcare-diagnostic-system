"""Deepchecks model-quality testing: data integrity, train/test validation and model evaluation suites.

    python -m ihds.mlops.quality             # exits 1 on a critical failure

Runs deepchecks' built-in `data_integrity` (full cleaned data), `train_test_validation` and `model_evaluation`
suites on the processed split and the trained model, saves the HTML reports to reports/deepchecks/ (gitignored) and a compact JSON summary to
reports/deepchecks_summary.json. A failed check named in params.yaml `evaluate.critical_checks`, or a test
ROC-AUC below `evaluate.min_test_roc_auc`, makes the stage exit 1 (which stops `dvc repro` and the Prefect
flow). Other failed checks (e.g. Simple Model Comparison, Feature Label Correlation Change) are recorded in
the summary but do not stop the pipeline. Checks deepchecks cannot run on this data (no datetime/index column,
regression- or boosting-only checks) are listed as `not_run`.

deepchecks 0.19.1 (the latest release) still references NumPy aliases removed in NumPy 2.0 (np.Inf, ...).
`_numpy2_aliases()` restores those aliases before import; it adds attributes only and changes no values.
"""
from __future__ import annotations

import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from ..config import FEATURES, TARGET
from .common import MLOPS_REPORTS, MODEL_PATH, load_params, load_splits, write_json

DC_DIR = MLOPS_REPORTS / "deepchecks"


def _numpy2_aliases() -> None:
    aliases = {"Inf": np.inf, "NaN": np.nan, "PINF": np.inf, "NINF": -np.inf, "infty": np.inf,
               "float_": np.float64, "complex_": np.complex128, "string_": np.bytes_, "unicode_": np.str_,
               "bool8": np.bool_, "object0": np.object_, "int0": np.intp, "uint0": np.uintp}
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _check_rows(result) -> list[dict]:
    rows = []
    for r in result.results:
        header = r.get_header() if hasattr(r, "get_header") else type(r).__name__
        if not hasattr(r, "conditions_results"):  # CheckFailure (check could not run)
            # e.g. "Dataset does not contain a datetime" or "irrelevant for classification tasks"
            rows.append({"check": header, "status": "not_run", "detail": str(getattr(r, "exception", ""))[:200]})
            continue
        conds = [{"name": c.name, "category": c.category.name, "details": str(c.details)[:300]}
                 for c in r.conditions_results]
        failed = [c for c in conds if c["category"] == "FAIL"]
        rows.append({"check": header, "status": "fail" if failed else ("warn" if any(
            c["category"] == "WARN" for c in conds) else "pass"), "conditions": conds})
    return rows


def _base_name(check: str) -> str:
    return check.split(" - ")[0].strip()


def run(train: pd.DataFrame | None = None, test: pd.DataFrame | None = None, pipe=None) -> dict:
    _numpy2_aliases()
    warnings.filterwarnings("ignore")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity, model_evaluation, train_test_validation

    p = load_params("evaluate")
    if train is None or test is None:
        train, test = load_splits()
    pipe = pipe if pipe is not None else joblib.load(MODEL_PATH)
    ds_train = Dataset(train[FEATURES + [TARGET]], label=TARGET, cat_features=[])
    ds_test = Dataset(test[FEATURES + [TARGET]], label=TARGET, cat_features=[])

    DC_DIR.mkdir(parents=True, exist_ok=True)
    full = pd.concat([train, test], ignore_index=True)[FEATURES + [TARGET]]
    suites = {"data_integrity": data_integrity().run(Dataset(full, label=TARGET, cat_features=[])),
              "train_test_validation": train_test_validation().run(ds_train, ds_test),
              "model_evaluation": model_evaluation().run(ds_train, ds_test, pipe)}
    summary, critical_failures = {}, []
    critical = set(p["critical_checks"])
    for name, res in suites.items():
        res.save_as_html(str(DC_DIR / f"{name}.html"), as_widget=False)
        rows = _check_rows(res)
        summary[name] = {"checks": len(rows), "passed": sum(r["status"] == "pass" for r in rows),
                         "warned": sum(r["status"] == "warn" for r in rows),
                         "failed": [r["check"] for r in rows if r["status"] == "fail"],
                         "not_run": [r["check"] for r in rows if r["status"] == "not_run"], "detail": rows}
        critical_failures += [r["check"] for r in rows if r["status"] == "fail"
                              and _base_name(r["check"]) in critical]

    from sklearn.metrics import roc_auc_score
    auc = round(float(roc_auc_score(test[TARGET], pipe.predict_proba(test[FEATURES])[:, 1])), 4)
    if auc < p["min_test_roc_auc"]:
        critical_failures.append(f"test ROC-AUC {auc} < {p['min_test_roc_auc']}")
    out = {"deepchecks_version": __import__("deepchecks").__version__, "test_roc_auc": auc,
           "critical_checks": sorted(critical), "critical_failures": critical_failures,
           "passed": not critical_failures, **summary}
    write_json(MLOPS_REPORTS / "deepchecks_summary.json", out)
    print({"passed": out["passed"], "critical_failures": critical_failures,
           **{k: {kk: v[kk] for kk in ("checks", "passed", "warned", "failed", "not_run")} for k, v in summary.items()}})
    return out


if __name__ == "__main__":
    sys.exit(0 if run()["passed"] else 1)
