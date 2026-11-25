"""Prefect 3 flow that orchestrates the MLOps stages locally.

    python -m ihds.mlops.flow

prepare -> validate (Great Expectations) -> train (MLflow) -> evaluate (Deepchecks) -> explain (SHAP + Fairlearn)
-> drift (Evidently) -> feature store (Feast). A failed validation or a critical Deepchecks failure raises, so
the flow ends in state Failed and later tasks do not run. No Prefect Cloud or server is needed: with no
PREFECT_API_URL set, Prefect starts a temporary local API for the run. Writes reports/flow_run.json.
"""
from __future__ import annotations

from prefect import flow, get_run_logger, task

from . import drift, explain, features, prepare, quality, train_tracked, validate
from .common import MLOPS_REPORTS, load_splits, write_json


class QualityGateError(RuntimeError):
    pass


@task(name="prepare")
def prepare_task() -> dict:
    return prepare.run()


@task(name="validate-gx")
def validate_task() -> dict:
    res = validate.run()
    if not res["success"]:
        raise QualityGateError(f"Great Expectations: {res['failed']} expectation(s) failed: {res['failures']}")
    return {k: res[k] for k in ("success", "evaluated", "passed", "failed")}


@task(name="train-mlflow")
def train_task() -> dict:
    train, test = load_splits()
    res = train_tracked.run(train, test)
    res.pop("pipe")
    return res


@task(name="evaluate-deepchecks")
def evaluate_task() -> dict:
    res = quality.run()
    if not res["passed"]:
        raise QualityGateError(f"Deepchecks critical failures: {res['critical_failures']}")
    return {k: res[k] for k in ("passed", "critical_failures", "test_roc_auc")}


@task(name="explain-shap-fairlearn")
def explain_task() -> dict:
    res = explain.run()
    f = res["fairness"]
    return {"shap_top3": [g["feature"] for g in res["shap"]["global"][:3]],
            "demographic_parity_difference": f["demographic_parity_difference"],
            "equalized_odds_difference": f["equalized_odds_difference"]}


@task(name="drift-evidently")
def drift_task() -> dict:
    return drift.run()


@task(name="feature-store-feast")
def feast_task() -> dict:
    res = features.run([0])
    return {"patient_ids": res["patient_ids"], "matches_source_parquet": res["matches_source_parquet"],
            "risk_probability": res.get("risk_probability")}


@flow(name="ihds-mlops-pipeline", log_prints=True)
def ihds_mlops_pipeline(with_feast: bool = True) -> dict:
    log = get_run_logger()
    out = {"prepare": prepare_task()}
    out["validate"] = validate_task()
    out["train"] = train_task()
    out["evaluate"] = evaluate_task()
    out["explain"] = explain_task()
    out["drift"] = drift_task()
    if with_feast:
        out["feast"] = feast_task()
    log.info("MLflow run %s, test ROC-AUC %s", out["train"]["mlflow_run_id"], out["train"]["test_roc_auc"])
    return out


def main() -> dict:
    state = ihds_mlops_pipeline(return_state=True)
    record = {"flow": "ihds-mlops-pipeline", "state": state.name}
    if state.is_completed():
        record["result"] = state.result()
    else:
        record["message"] = state.message
    write_json(MLOPS_REPORTS / "flow_run.json", record)
    print({"flow_state": record["state"]})
    if not state.is_completed():
        raise SystemExit(1)
    return record


if __name__ == "__main__":
    main()
