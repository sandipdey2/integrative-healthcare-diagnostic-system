"""Assessment audit log: SQLite unit tests, plus a Postgres integration test skipped without DATABASE_URL."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from ihds import audit_log
from ihds.config import FEATURES

REC = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 72, "SkinThickness": 30, "Insulin": 120,
       "BMI": 34.0, "DiabetesPedigreeFunction": 0.5, "Age": 45, "question": "please ignore this",
       "notes": "not a feature"}


def _res(**kw):
    defaults = dict(status="ok", probability=0.61, band="high", draft_source="template", revisions=0,
                    escalated=False, retrieval="chroma", langsmith_run_id="",
                    review={"approved": True, "grounding_score": 0.55}, draft="this must not be stored")
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_sqlite_configure_records_and_lists(tmp_path, monkeypatch):
    monkeypatch.setenv("IHDS_AUDIT_SQLITE", str(tmp_path / "audit.db"))
    audit_log.reset()
    info = audit_log.configure()
    assert info["backend"] == "sqlite (local)"
    assert (tmp_path / "audit.db").exists()
    assert audit_log.record(_res(), REC, workflow_route="in-process") is True
    assert audit_log.count() == 1
    rows = audit_log.recent(20)
    assert len(rows) == 1
    row = rows[0]
    assert row["risk_band"] == "high" and row["reviewer_status"] == "approved"
    assert row["draft_source"] == "template" and row["workflow_route"] == "in-process"
    assert row["retrieval_method"] == "chroma" and row["blocked"] is False
    assert set(row["features"]) == set(FEATURES)
    assert row["features"]["Glucose"] == 150 and row["features"]["Age"] == 45
    assert "question" not in row["features"] and "notes" not in row["features"]
    assert "please ignore" not in str(row) and "this must not be stored" not in str(row)
    assert "question" not in row
    st = audit_log.status()
    assert st["backend"] == "sqlite (local)" and st["rows"] == 1 and st["writes_ok"] >= 1


def test_blocked_and_rejected_status_and_order():
    audit_log.configure()
    assert audit_log.record(_res(status="blocked", probability=None, band=None, review={}), REC,
                            workflow_route="rest-api")
    assert audit_log.record(_res(review={"approved": False}, draft_source="llm", revisions=2, escalated=True,
                                 langsmith_run_id="abc-123"), REC, workflow_route="embedded-api")
    rows = audit_log.recent(10)
    assert rows[0]["reviewer_status"] == "rejected" and rows[0]["escalated"] is True
    assert rows[0]["langsmith_run_id"] == "abc-123" and rows[0]["workflow_route"] == "embedded-api"
    assert rows[1]["blocked"] is True and rows[1]["reviewer_status"] == "n/a"


def test_write_failure_is_counted_and_does_not_raise():
    audit_log.configure()
    saved = audit_log._Session
    audit_log._Session = None  # force the write path to fail after _ensure
    try:
        assert audit_log.record(_res(), REC) is False
    finally:
        audit_log._Session = saved
    assert audit_log.status()["writes_failed"] >= 1


def test_redact_strips_credentials():
    url = "postgresql://ihds_app:super-secret@ep-example.neon.tech/ihds?sslmode=require"
    out = audit_log.redact(f"could not connect to {url}")
    assert "super-secret" not in out and "ihds_app" not in out
    assert "neon.tech" in out and "***@" in out
    assert "super-secret" not in audit_log.redact("OperationalError: " + url)


def test_backend_label_detects_neon():
    assert audit_log._backend_for("postgresql://u:p@ep-foo.us-west-2.aws.neon.tech/ihds") == "postgres (neon)"
    assert audit_log._backend_for("postgresql://ihds:ihds@postgres:5432/ihds") == "postgres"
    assert audit_log._backend_for("sqlite:////tmp/audit.db") == "sqlite (local)"


def test_configure_reads_streamlit_style_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("IHDS_AUDIT_SQLITE", raising=False)
    path = tmp_path / "from-secret.db"
    secrets = {"database": {"url": f"sqlite:///{path}"}}
    audit_log.reset()
    info = audit_log.configure(secrets=secrets)
    assert info["backend"] == "sqlite (local)"
    assert path.exists() or audit_log.record(_res(), REC)
    # secret URL must not have been copied into the process environment
    assert "DATABASE_URL" not in os.environ


def test_api_recent_endpoint_and_metrics(system):
    from fastapi.testclient import TestClient
    import api.main as api_main

    api_main.STATE["system"] = system
    client = TestClient(api_main.app)
    rec = {k: REC[k] for k in FEATURES}
    r = client.post("/assess", json={"record": rec})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    body = client.get("/assessments/recent?limit=5").json()
    assert body["backend"] == "sqlite (local)" and body["count"] >= 1
    item = body["items"][0]
    assert item["risk_band"] in ("low", "moderate", "high")
    assert set(item["features"]) == set(FEATURES)
    assert item["workflow_route"] in ("rest-api", "embedded-api")
    metrics = client.get("/metrics").text
    assert "ihds_audit_writes_total" in metrics and 'result="ok"' in metrics


@pytest.mark.postgres
def test_postgres_roundtrip_when_database_url_set():
    """Live write/read against DATABASE_URL (Neon). Skipped when the env var is unset."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        pytest.skip("DATABASE_URL not set")
    audit_log.reset()
    info = audit_log.configure(url=url)
    if info["backend"] not in ("postgres (neon)", "postgres"):
        pytest.skip(f"Postgres unreachable from this host; fell back to {info['backend']}")
    before = audit_log.count()
    ok = audit_log.record(_res(draft_source="template", band="moderate", probability=0.42),
                          REC, workflow_route="pytest-neon")
    assert ok is True
    assert audit_log.count() == before + 1
    rows = audit_log.recent(5)
    assert rows and rows[0]["workflow_route"] == "pytest-neon"
    assert rows[0]["features"]["Glucose"] == 150
    assert "super-secret" not in str(rows[0])
