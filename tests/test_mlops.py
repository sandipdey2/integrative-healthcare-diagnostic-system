"""Optional MLOps layer (ihds/mlops, requirements-mlops.txt). Each test skips cleanly when its tool is not
installed, so the runtime environment (requirements-dev.txt) still runs the rest of the suite."""
import importlib.util
import json
import re
import shutil

import pandas as pd
import pytest

from ihds import data
from ihds.config import FEATURES, ROOT

yaml = pytest.importorskip("yaml")


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


needs = {m: pytest.mark.skipif(not _has(m), reason=f"{m} not installed (pip install -r requirements-mlops.txt)")
         for m in ("great_expectations", "mlflow", "deepchecks", "shap", "fairlearn", "feast", "prefect")}


@pytest.fixture
def reports_tmp(tmp_path, monkeypatch):
    """Point every MLOps module's report / model paths at a temp dir."""
    from ihds.mlops import common

    out = tmp_path / "reports"
    monkeypatch.setattr(common, "MLOPS_REPORTS", out)
    for name in ("validate", "train_tracked", "quality", "explain", "drift", "features"):
        if name in ("quality",) and not _has("deepchecks"):
            continue
        mod = importlib.import_module(f"ihds.mlops.{name}")
        if hasattr(mod, "MLOPS_REPORTS"):
            monkeypatch.setattr(mod, "MLOPS_REPORTS", out)
        if hasattr(mod, "MODEL_PATH"):
            monkeypatch.setattr(mod, "MODEL_PATH", tmp_path / "models" / "risk_model.joblib")
    return out


# ------------------------------------------------------------------ config files (no optional deps)
def test_params_and_dvc_pipeline_are_consistent():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    dvc = yaml.safe_load((ROOT / "dvc.yaml").read_text())
    assert list(dvc["stages"]) == ["prepare", "validate", "train", "evaluate", "cm_analysis", "explain", "drift", "features"]
    for name, st in dvc["stages"].items():
        mod = re.match(r"python -m ([\w.]+)", st["cmd"]).group(1)
        assert (ROOT / (mod.replace(".", "/") + ".py")).exists(), name
        for dep in st.get("deps", []):
            produced = any(dep in [next(iter(o)) if isinstance(o, dict) else o
                                   for o in s.get("outs", []) + s.get("metrics", [])]
                           for s in dvc["stages"].values())
            assert produced or (ROOT / dep).exists(), f"{name}: {dep}"
        for p in st.get("params", []):
            section, _, key = p.partition(".")
            assert section in params and (not key or key in params[section])
    assert (ROOT / ".dvc" / "config").exists()


def test_prometheus_grafana_configs_reference_real_metrics():
    prom = yaml.safe_load((ROOT / "monitoring/prometheus/prometheus.yml").read_text())
    assert prom["scrape_configs"][0]["static_configs"][0]["targets"] == ["api:8000"]
    compose = yaml.safe_load((ROOT / "docker-compose.monitoring.yml").read_text())
    assert set(compose["services"]) == {"api", "prometheus", "grafana", "postgres", "loki", "alloy"}  # loki + alloy: logs (tests/test_logging.py)
    assert all(p.startswith("127.0.0.1:") for s in compose["services"].values() for p in s.get("ports", []))
    assert all(s["networks"] == ["ihds-monitoring"] for s in compose["services"].values())
    assert compose["services"]["prometheus"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert compose["services"]["api"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert compose["services"]["postgres"]["image"].startswith("postgres:17")
    assert "postgres:5432" in compose["services"]["api"]["environment"]["DATABASE_URL"]
    assert "0.0.0.0" in compose["services"]["api"]["command"]
    ds = yaml.safe_load((ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml").read_text())
    assert ds["datasources"][0]["url"] == "${PROMETHEUS_URL}"
    dash = json.loads((ROOT / "monitoring/grafana/dashboards/ihds_api.json").read_text())
    exprs = [t["expr"] for p in dash["panels"] for t in p["targets"]]
    # datasource is a dashboard variable (importable into Grafana Cloud), never a hard-coded uid
    var = dash["templating"]["list"][0]
    assert var["name"] == "datasource" and var["type"] == "datasource" and var["query"] == "prometheus"
    assert all(p["datasource"]["uid"] == "${datasource}" for p in dash["panels"])
    assert all(t["datasource"]["uid"] == "${datasource}" for p in dash["panels"] for t in p["targets"])
    used = set(re.findall(r"ihds_[a-z_]+", " ".join(exprs)))
    defined = {"ihds_http_requests_total", "ihds_predictions_total", "ihds_assess_total", "ihds_escalations_total",
               "ihds_reviewer_revisions_total", "ihds_assess_duration_seconds_bucket", "ihds_app_up"}
    assert [v["name"] for v in dash["templating"]["list"]] == ["datasource", "service"]
    assert all('service=~"$service"' in e for e in exprs)
    assert used and used <= defined
    if _has("prometheus_client") and _has("fastapi") and _has("langgraph"):  # runtime env only
        import api.main as api_main  # noqa: F401  (registers the metrics)
        from prometheus_client import REGISTRY

        # compare metric families, not samples: a labelled counter has no samples until it is first incremented,
        # so a sample-based check depended on test order
        families = {m.name: m.type for m in REGISTRY.collect()}
        for name in used:
            base = re.sub(r"_(total|bucket)$", "", name)
            assert base in families, name
            kind = "histogram" if name.endswith("_bucket") else "gauge" if name == "ihds_app_up" else "counter"
            assert families[base] == kind, name


def test_ci_workflow_example_has_required_jobs():
    wf = yaml.safe_load((ROOT / "docs/ci.yml.example").read_text())
    jobs = wf["jobs"]
    assert {"lint", "test", "data-validation", "docker"} <= set(jobs)
    steps = " ".join(str(s.get("run", "")) for j in jobs.values() for s in j.get("steps", []))
    assert "ruff check" in steps and "pytest" in steps and "ihds.mlops.validate" in steps and "docker build" in steps


# ------------------------------------------------------------------ Great Expectations
@needs["great_expectations"]
def test_gx_suite_passes_on_shipped_data_and_catches_bad_data():
    from ihds.mlops import validate

    params = {"range_mostly": 0.99, "min_rows": 700, "max_rows": 1000}
    df = data.clean(pd.read_csv(ROOT / "data" / "pima_diabetes.csv"))
    ok = validate.validate_frame(df, params)
    assert ok["success"] and ok["evaluated"] == ok["passed"] == 30
    bad = df.copy()
    bad.loc[: 49, "Glucose"] = 900          # 50 impossible values (6.5% > 1% allowed)
    bad.loc[: 99, "Age"] = None             # 100 missing ages (Age must be complete)
    res = validate.validate_frame(bad, params)
    failed = {(f["expectation"], f["kwargs"]["column"]) for f in res["failures"]}
    assert not res["success"]
    assert ("expect_column_values_to_be_between", "Glucose") in failed
    assert ("expect_column_values_to_not_be_null", "Age") in failed


# ------------------------------------------------------------------ MLflow
@needs["mlflow"]
def test_mlflow_tracked_training_logs_params_metrics_and_model(tmp_path, monkeypatch, split, reports_tmp):
    import mlflow

    from ihds.mlops import train_tracked

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(train_tracked, "MLRUNS_DIR", tmp_path / "mlruns")
    train, test = split
    res = train_tracked.run(train, test)
    run = mlflow.get_run(res["mlflow_run_id"])
    assert run.data.params["class_weight"] == "balanced" and run.data.params["n_train"] == str(len(train))
    assert run.data.metrics["test_roc_auc"] == pytest.approx(res["test_roc_auc"]) and res["test_roc_auc"] > 0.75
    loaded = mlflow.sklearn.load_model(res["model_uri"])
    x = test[FEATURES].astype("float64")
    assert (loaded.predict_proba(x)[:, 1] == pytest.approx(res["pipe"].predict_proba(x)[:, 1]))
    assert json.loads((reports_tmp / "train_metrics.json").read_text())["mlflow_run_id"] == res["mlflow_run_id"]


# ------------------------------------------------------------------ Deepchecks
@needs["deepchecks"]
def test_deepchecks_passes_real_split_and_fails_on_leakage(split, pipe, reports_tmp, monkeypatch):
    from ihds.mlops import quality

    monkeypatch.setattr(quality, "DC_DIR", reports_tmp / "deepchecks")
    train, test = split
    ok = quality.run(train, test, pipe)
    assert ok["passed"] and ok["critical_failures"] == []
    assert (reports_tmp / "deepchecks" / "model_evaluation.html").stat().st_size > 100_000
    leaked = pd.concat([test.iloc[:60], train.iloc[:94]], ignore_index=True)  # 94 training rows in "test"
    bad = quality.run(train, leaked, pipe)
    assert not bad["passed"] and "Train Test Samples Mix" in bad["critical_failures"]


# ------------------------------------------------------------------ SHAP + Fairlearn
@needs["shap"]
def test_shap_values_are_additive_and_glucose_ranks_first(split, pipe, reports_tmp):
    import numpy as np

    from ihds.mlops import explain

    reports_tmp.mkdir(parents=True, exist_ok=True)
    train, test = split
    out = explain.shap_explain(pipe, train, test, background=100, local_examples=3, plot=False)
    assert out["global"][0]["feature"] == "Glucose" and len(out["global"]) == len(FEATURES)
    for ex in out["local"]:  # LinearExplainer is additive: base + sum(SHAP) = model log-odds
        z = explain._transform(pipe, test.iloc[[ex["test_row"]]])
        logit = float(pipe.named_steps["clf"].decision_function(z)[0])
        assert ex["base_value_logodds"] + ex["sum_shap_logodds"] == pytest.approx(logit, abs=1e-3)
        assert ex["prediction_proba"] == pytest.approx(1 / (1 + np.exp(-logit)), abs=1e-4)
        assert len(ex["top_features"]) == 3


@needs["fairlearn"]
def test_fairlearn_age_band_audit(split, pipe):
    from ihds.mlops import explain

    test = split[1]
    f = explain.fairness_audit(pipe, test, [21, 30, 45, 120], ["21-29", "30-44", "45+"])
    assert sum(g["count"] for g in f["by_group"]) == len(test)
    assert 0 <= f["demographic_parity_difference"] <= 1 and 0 <= f["equalized_odds_difference"] <= 1
    rates = [g["selection_rate"] for g in f["by_group"]]
    assert f["demographic_parity_difference"] == pytest.approx(max(rates) - min(rates), abs=1e-3)
    assert "Age" in f["sensitive_attribute"]


# ------------------------------------------------------------------ Feast
@needs["feast"]
def test_feast_apply_materialize_and_online_fetch(tmp_path, monkeypatch, reports_tmp):
    from ihds.mlops import features, prepare

    repo = tmp_path / "feature_repo"
    (repo / "data").mkdir(parents=True)
    for f in ("feature_store.yaml", "features.py"):
        shutil.copy(ROOT / "feature_repo" / f, repo / f)
    parquet = repo / "data" / "patient_features.parquet"
    src = prepare.feast_frame(data.load_clean(), "2026-01-01T00:00:00+00:00")
    src.to_parquet(parquet, index=False)
    monkeypatch.setattr(features, "FEATURE_REPO", repo)
    monkeypatch.setattr(features, "FEAST_PARQUET", parquet)
    out = features.run([0, 5])
    assert out["feature_views"] == ["patient_clinical_features"] and out["matches_source_parquet"]
    row0 = out["features"][0]
    assert row0["patient_id"] == 0 and row0["Glucose"] == 148.0 and row0["Insulin"] is None  # raw 0 -> missing


# ------------------------------------------------------------------ Prefect
@needs["prefect"]
def test_prefect_flow_completes_and_stops_on_failed_validation(monkeypatch):
    from prefect.testing.utilities import prefect_test_harness

    from ihds.mlops import drift, explain, features, flow, prepare, quality, train_tracked, validate

    calls = []
    monkeypatch.setattr(prepare, "run", lambda: calls.append("prepare") or {"rows": 768})
    monkeypatch.setattr(validate, "run", lambda: calls.append("validate") or
                        {"success": True, "evaluated": 30, "passed": 30, "failed": 0, "failures": []})
    monkeypatch.setattr(flow, "load_splits", lambda: (None, None))
    monkeypatch.setattr(train_tracked, "run", lambda tr, te: calls.append("train") or
                        {"mlflow_run_id": "x", "test_roc_auc": 0.8, "pipe": object()})
    monkeypatch.setattr(quality, "run", lambda: calls.append("evaluate") or
                        {"passed": True, "critical_failures": [], "test_roc_auc": 0.8})
    monkeypatch.setattr(explain, "run", lambda: calls.append("explain") or {
        "shap": {"global": [{"feature": "Glucose"}]},
        "fairness": {"demographic_parity_difference": 0.1, "equalized_odds_difference": 0.2}})
    monkeypatch.setattr(drift, "run", lambda: calls.append("drift") or {})
    monkeypatch.setattr(features, "run", lambda ids: calls.append("feast") or
                        {"patient_ids": ids, "matches_source_parquet": True})
    with prefect_test_harness():  # throwaway local Prefect API + DB for the test
        ok = flow.ihds_mlops_pipeline(return_state=True)
        calls_ok = list(calls)
        calls.clear()
        monkeypatch.setattr(validate, "run", lambda: calls.append("validate") or
                            {"success": False, "evaluated": 30, "passed": 29, "failed": 1, "failures": [{"x": 1}]})
        bad = flow.ihds_mlops_pipeline(return_state=True)
    assert ok.is_completed()
    assert calls_ok == ["prepare", "validate", "train", "evaluate", "explain", "drift", "feast"]
    assert bad.is_failed() and calls == ["prepare", "validate"]



# ------------------------------------------------------------------ Prefect deployment (schedule + retries)
def _fake_stages(monkeypatch, calls):
    """Replace every stage with a fake that records its name (same fakes as the flow test above)."""
    from ihds.mlops import drift, explain, features, flow, prepare, quality, train_tracked, validate

    monkeypatch.setattr(prepare, "run", lambda: calls.append("prepare") or {"rows": 768})
    monkeypatch.setattr(validate, "run", lambda: calls.append("validate") or
                        {"success": True, "evaluated": 30, "passed": 30, "failed": 0, "failures": []})
    monkeypatch.setattr(flow, "load_splits", lambda: (None, None))
    monkeypatch.setattr(train_tracked, "run", lambda tr, te: calls.append("train") or
                        {"mlflow_run_id": "x", "test_roc_auc": 0.8, "pipe": object()})
    monkeypatch.setattr(quality, "run", lambda: calls.append("evaluate") or
                        {"passed": True, "critical_failures": [], "test_roc_auc": 0.8})
    monkeypatch.setattr(explain, "run", lambda: calls.append("explain") or {
        "shap": {"global": [{"feature": "Glucose"}]},
        "fairness": {"demographic_parity_difference": 0.1, "equalized_odds_difference": 0.2}})
    monkeypatch.setattr(drift, "run", lambda: calls.append("drift") or {})
    monkeypatch.setattr(features, "run", lambda ids: calls.append("feast") or
                        {"patient_ids": ids, "matches_source_parquet": True})


def _no_retry_wait(monkeypatch):
    """Keep the retry count and condition, but do not wait 10 s between attempts in a test."""
    from ihds.mlops import flow

    for name in ("prepare_task", "validate_task", "train_task", "evaluate_task", "explain_task", "drift_task",
                 "feast_task"):
        monkeypatch.setattr(flow, name, getattr(flow, name).with_options(retry_delay_seconds=0))


@needs["prefect"]
def test_deployment_object_has_name_schedule_parameters_and_tags():
    from prefect.flows import EntrypointType

    from ihds.mlops import deploy
    from ihds.mlops.common import load_params

    cfg = load_params("deploy")
    dep = deploy.build_deployment()
    info = deploy.describe(dep)
    assert info["deployment"] == "ihds-mlops-pipeline/" + cfg["name"] == "ihds-mlops-pipeline/weekly-local"
    assert (info["cron"], info["timezone"]) == ("0 6 * * 1", "America/Los_Angeles")
    assert info["parameters"] == {"with_feast": True}
    assert set(info["tags"]) == {"ihds", "mlops", "local"}
    assert info["paused"] is False
    assert dep.entrypoint_type == EntrypointType.MODULE_PATH  # flow.py uses relative imports
    # explicit arguments override params.yaml
    other = deploy.describe(deploy.build_deployment("nightly", "30 2 * * *", "UTC", {"with_feast": False}, ["x"], True))
    assert other == {"deployment": "ihds-mlops-pipeline/nightly", "cron": "30 2 * * *", "timezone": "UTC",
                     "parameters": {"with_feast": False}, "tags": ["x"], "paused": True}


@needs["prefect"]
def test_deploy_cli_dry_run_prints_the_deployment_and_registers_nothing():
    import contextlib
    import io
    import json

    from ihds.mlops import deploy

    buf = io.StringIO()  # not capsys: it swaps stderr, which breaks Prefect's console log handler in later tests
    with contextlib.redirect_stdout(buf):
        assert deploy.main(["--dry-run", "--cron", "0 7 * * 2", "--no-feast", "--paused"]) == 0
    out = json.loads(buf.getvalue())
    assert out["cron"] == "0 7 * * 2" and out["parameters"] == {"with_feast": False} and out["paused"] is True


@needs["prefect"]
def test_every_task_retries_but_not_on_a_quality_gate_failure():
    from ihds.mlops import flow

    tasks = (flow.prepare_task, flow.validate_task, flow.train_task, flow.evaluate_task, flow.explain_task,
             flow.drift_task, flow.feast_task)
    assert all(t.retries == flow.TASK_RETRIES == 2 for t in tasks)
    assert all(t.retry_condition_fn is flow.retry_unless_gate_failure for t in tasks)

    class Failed:
        def __init__(self, exc):
            self.exc = exc

        def result(self):
            raise self.exc

    class Ok:
        def result(self):
            return {}

    assert flow.retry_unless_gate_failure(None, None, Failed(OSError("locked"))) is True
    assert flow.retry_unless_gate_failure(None, None, Failed(flow.QualityGateError("x"))) is False
    assert flow.retry_unless_gate_failure(None, None, Ok()) is False


@needs["prefect"]
def test_flaky_stage_is_retried_but_failed_validation_fails_fast(monkeypatch):
    from prefect.testing.utilities import prefect_test_harness

    from ihds.mlops import flow, prepare, validate

    calls = []
    _fake_stages(monkeypatch, calls)
    _no_retry_wait(monkeypatch)
    attempts = {"n": 0}

    def flaky_prepare():
        calls.append("prepare")
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("transient")
        return {"rows": 768}

    monkeypatch.setattr(prepare, "run", flaky_prepare)
    with prefect_test_harness():
        ok = flow.ihds_mlops_pipeline(return_state=True)
        calls_ok = list(calls)
        calls.clear()
        monkeypatch.setattr(prepare, "run", lambda: calls.append("prepare") or {"rows": 768})
        monkeypatch.setattr(validate, "run", lambda: calls.append("validate") or
                            {"success": False, "evaluated": 30, "passed": 29, "failed": 1, "failures": [{"x": 1}]})
        bad = flow.ihds_mlops_pipeline(return_state=True)
    assert ok.is_completed() and calls_ok[:3] == ["prepare", "prepare", "validate"]  # prepare retried once
    assert bad.is_failed() and calls == ["prepare", "validate"]  # gate failure: validate ran once, nothing after it


@needs["prefect"]
def test_deployment_registers_and_a_triggered_run_completes(monkeypatch):
    """Register the deployment on a throwaway API, trigger it, and execute the created run in this process.

    `flow.serve()` itself runs each flow run in a subprocess, where the monkeypatched fake stages would not
    exist, so the run is executed in-process with the flow engine. That is the same code path (deployment ->
    flow run -> flow function); the real subprocess path was checked by hand (docs/MLOPS.md)."""
    import asyncio

    from prefect import get_client
    from prefect.deployments import run_deployment
    from prefect.flow_engine import run_flow
    from prefect.testing.utilities import prefect_test_harness

    from ihds.mlops import deploy, flow

    calls = []
    _fake_stages(monkeypatch, calls)
    with prefect_test_harness():
        dep = deploy.build_deployment()
        dep_id = dep.apply()
        fr = run_deployment("ihds-mlops-pipeline/weekly-local", timeout=0)

        async def read():
            async with get_client() as c:
                return await c.read_deployment(dep_id)

        stored = asyncio.run(read())
        assert fr.deployment_id == stored.id and fr.parameters == {"with_feast": True}
        state = run_flow(flow.ihds_mlops_pipeline, flow_run=fr, parameters=fr.parameters, return_type="state")
    assert stored.name == "weekly-local" and set(stored.tags) == {"ihds", "mlops", "local"}
    assert [(s.schedule.cron, s.schedule.timezone) for s in stored.schedules] == [("0 6 * * 1", "America/Los_Angeles")]
    assert state.is_completed()
    assert calls == ["prepare", "validate", "train", "evaluate", "explain", "drift", "feast"]
