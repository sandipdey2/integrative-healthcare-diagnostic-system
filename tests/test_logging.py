"""Structured JSON logging (ihds/logging_config.py + api/main.py middleware) and the Loki wiring files.

No Docker and no Loki needed: the formatter, the request/assessment log lines and the config files are checked
offline. A live Loki run is a manual step (docs/MLOPS.md, "Logs: Loki + Grafana Alloy").
"""
import io
import json
import logging
import re

import pytest
import yaml
from fastapi.testclient import TestClient

from ihds import logging_config as lc
from ihds.config import FEATURES, ROOT

SECRET_QUESTION = "ZEBRA-SECRET-QUESTION how do I lower my numbers"
REC = {"Pregnancies": 2, "Glucose": 151.25, "BloodPressure": 73, "SkinThickness": 31, "Insulin": 123.5, "BMI": 34.125,
       "DiabetesPedigreeFunction": 0.5123, "Age": 45}


def _logger_to_buffer():
    """Attach a JSON handler to the ihds.api logger; returns (buffer, cleanup)."""
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(lc.JsonFormatter())
    log = logging.getLogger(lc.LOGGER_NAME)
    log.addHandler(h)
    old = log.level
    log.setLevel(logging.INFO)
    return buf, lambda: (log.removeHandler(h), log.setLevel(old))


def _lines(buf):
    return [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]


@pytest.fixture
def client(system, monkeypatch):
    import api.main as api_main

    monkeypatch.setitem(api_main.STATE, "system", system)
    return TestClient(api_main.app)


@pytest.fixture
def logbuf():
    buf, cleanup = _logger_to_buffer()
    yield buf
    cleanup()


# ---------------------------------------------------------------- formatter
def test_formatter_emits_valid_json_with_standard_fields():
    rec = logging.LogRecord("ihds.api", logging.WARNING, __file__, 1, "request", None, None)
    rec.event, rec.route, rec.status, rec.latency_ms = "request", "/assess", 422, 3.5
    out = json.loads(lc.JsonFormatter().format(rec))
    assert {"timestamp", "level", "logger", "message"} <= out.keys()
    assert out["level"] == "warning" and out["logger"] == "ihds.api" and out["message"] == "request"
    assert out["route"] == "/assess" and out["status"] == 422 and out["latency_ms"] == 3.5
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00", out["timestamp"])


def test_formatter_drops_unknown_fields_and_exception_text():
    """A log call cannot smuggle data in through extra=; exceptions are reported by class name only."""
    rec = logging.LogRecord("ihds.api", logging.ERROR, __file__, 1, "request", None, None)
    rec.Glucose, rec.question, rec.prompt, rec.api_key = 151, SECRET_QUESTION, "system prompt", "sk-secret"
    try:
        raise ValueError("Glucose=151 is bad")
    except ValueError:
        import sys
        rec.exc_info = sys.exc_info()
    line = lc.JsonFormatter().format(rec)
    out = json.loads(line)
    assert out["exc_type"] == "ValueError"
    for needle in ("Glucose", "151", "ZEBRA", "system prompt", "sk-secret", "Traceback", "is bad"):
        assert needle not in line


def test_allowed_fields_contain_no_patient_or_text_fields():
    banned = {f.lower() for f in FEATURES} | {"question", "prompt", "draft", "answer", "record", "probability",
                                               "guard_reasons", "api_key", "token", "password"}
    assert banned.isdisjoint({f.lower() for f in lc.ALLOWED_FIELDS})


def test_request_id_accepts_only_safe_values():
    assert lc.new_request_id("abc-123_X.y") == "abc-123_X.y"
    for bad in ("", None, "a b", "x" * 65, 'a"b', "a\nb", "{json}"):
        got = lc.new_request_id(bad)
        assert re.fullmatch(r"[0-9a-f]{16}", got)


def test_configure_writes_jsonl_file_and_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "ihds.jsonl"
    # configure() works on the fixed logger name, so reset that logger for the test and restore it afterwards
    base = logging.getLogger(lc.LOGGER_NAME)
    saved = (list(base.handlers), base.level, base.propagate, getattr(base, "_ihds_configured", False))
    base.handlers.clear()
    base._ihds_configured = False
    monkeypatch.setenv("IHDS_LOG_FILE", str(path))
    monkeypatch.setenv("IHDS_LOG_JSON", "0")
    try:
        lc.configure()
        n = len(base.handlers)
        lc.configure()
        assert len(base.handlers) == n == 1
        base.info("hello", extra={"event": "test"})
        for h in base.handlers:
            h.flush()
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert rows[0]["message"] == "hello" and rows[0]["event"] == "test"
    finally:
        for h in list(base.handlers):
            h.close()
        base.handlers[:] = saved[0]
        base.setLevel(saved[1])
        base.propagate = saved[2]
        base._ihds_configured = saved[3]


def test_unwritable_log_file_does_not_stop_the_api(tmp_path, monkeypatch, capsys):
    base = logging.getLogger(lc.LOGGER_NAME)
    saved = (list(base.handlers), base.level, base.propagate, getattr(base, "_ihds_configured", False))
    base.handlers.clear()
    base._ihds_configured = False
    blocker = tmp_path / "file"
    blocker.write_text("x")  # a file where a directory is needed
    monkeypatch.setenv("IHDS_LOG_FILE", str(blocker / "ihds.jsonl"))
    monkeypatch.setenv("IHDS_LOG_JSON", "0")
    try:
        lc.configure()
        assert base.handlers == []
        assert "cannot write" in capsys.readouterr().err
    finally:
        base.handlers[:] = saved[0]
        base.setLevel(saved[1])
        base.propagate = saved[2]
        base._ihds_configured = saved[3]


# ---------------------------------------------------------------- API log lines
def test_request_log_line_has_id_route_status_latency(client, logbuf):
    r = client.get("/health", headers={"X-Request-ID": "test-req-42"})
    assert r.status_code == 200 and r.headers["x-request-id"] == "test-req-42"
    rows = [x for x in _lines(logbuf) if x.get("event") == "request"]
    assert len(rows) == 1
    row = rows[0]
    assert row["request_id"] == "test-req-42" and row["route"] == "/health" and row["status"] == 200
    assert row["method"] == "GET" and isinstance(row["latency_ms"], float) and row["latency_ms"] >= 0
    assert row["level"] == "info"


def test_request_id_is_generated_and_returned_when_missing(client, logbuf):
    r = client.get("/health")
    rid = r.headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{16}", rid)
    assert _lines(logbuf)[0]["request_id"] == rid


def test_unsafe_incoming_request_id_is_replaced(client, logbuf):
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["x-request-id"] != "bad id with spaces"
    assert "bad id" not in logbuf.getvalue()


def test_levels_follow_status_and_unmatched_routes_use_a_constant(client, logbuf):
    client.get("/no-such-path/12345?x=1")
    client.post("/assess", json={"record": {"Glucose": "abc"}})  # 422 from pydantic
    rows = [x for x in _lines(logbuf) if x.get("event") == "request"]
    assert [(x["route"], x["status"], x["level"]) for x in rows] == [("unmatched", 404, "warning"),
                                                                      ("/assess", 422, "warning")]
    assert "12345" not in logbuf.getvalue()  # raw path and query string are not logged


def test_metrics_scrapes_are_not_logged(client, logbuf):
    assert client.get("/metrics").status_code == 200
    assert logbuf.getvalue() == ""


def test_assess_logs_request_and_assessment_lines_with_the_same_id(client, logbuf):
    r = client.post("/assess", json={"record": REC, "question": "What drives my risk?"},
                    headers={"X-Request-ID": "assess-1"})
    assert r.status_code == 200
    body = r.json()
    rows = _lines(logbuf)
    a = next(x for x in rows if x.get("event") == "assessment")
    q = next(x for x in rows if x.get("event") == "request")
    assert a["request_id"] == q["request_id"] == "assess-1"
    assert q["route"] == "/assess" and q["status"] == 200
    assert a["band"] == body["band"] and a["draft_source"] == body["draft_source"]
    assert a["escalated"] is body["escalated"] and a["revisions"] == body["revisions"]
    assert a["assess_status"] == "ok" and a["blocked"] is False
    assert a["trace_nodes"][0] == "safety_guard" and a["trace_nodes"][-1] == "finalize"
    assert all(re.fullmatch(r"[a-z_]+", n) for n in a["trace_nodes"])  # node names only, no trace text


def test_blocked_assessment_logs_a_count_not_the_reasons(client, logbuf):
    bad = dict(REC, Glucose=999)
    r = client.post("/assess", json={"record": bad, "question": "Should I stop my medication?"})
    assert r.json()["status"] == "blocked"
    a = next(x for x in _lines(logbuf) if x.get("event") == "assessment")
    assert a["blocked"] is True and a["guard_reason_count"] >= 2 and a["band"] == "none"
    assert "999" not in logbuf.getvalue() and "medication" not in logbuf.getvalue()


def test_no_patient_values_question_or_draft_text_in_logs(client, logbuf):
    r = client.post("/assess", json={"record": REC, "question": SECRET_QUESTION})
    assert r.status_code == 200
    client.post("/assess", json={"record": dict(REC, Glucose=999), "question": SECRET_QUESTION})
    client.get("/assessments/recent?limit=3")
    text = logbuf.getvalue()
    assert text.strip()
    draft = r.json()["draft"]
    needles = ["ZEBRA", "SECRET-QUESTION", "lower my numbers", "151.25", "34.125", "123.5", "0.5123", '"Glucose"',
               "Insulin", "BMI", "DiabetesPedigree", draft[:40], "OPENAI_API_KEY", "sk-", "voc-", "postgresql://",
               "probability"]
    for n in needles:
        assert n not in text, n
    for row in _lines(logbuf):
        assert set(row) <= {"timestamp", "level", "logger", "message", *lc.ALLOWED_FIELDS}


def test_unhandled_exception_is_logged_as_error_by_class_only(system, monkeypatch, logbuf):
    import api.main as api_main

    monkeypatch.setitem(api_main.STATE, "system", system)

    def boom(*a, **k):
        raise RuntimeError("Glucose=151 secret detail")

    monkeypatch.setattr(api_main.agents, "run", boom)
    c = TestClient(api_main.app, raise_server_exceptions=False)
    assert c.post("/assess", json={"record": REC}).status_code == 500
    row = next(x for x in _lines(logbuf) if x.get("event") == "request")
    assert row["level"] == "error" and row["status"] == 500 and row["exc_type"] == "RuntimeError"
    assert "secret detail" not in logbuf.getvalue() and "151" not in logbuf.getvalue()


# ---------------------------------------------------------------- Loki wiring files
COMPOSE = ROOT / "docker-compose.monitoring.yml"
DASH = ROOT / "monitoring/grafana/dashboards/ihds_logs.json"
LOKI_CFG = ROOT / "monitoring/loki/loki-config.yaml"
ALLOY_CFG = ROOT / "monitoring/loki/alloy-config.alloy"
DATASOURCE = ROOT / "monitoring/grafana/provisioning/datasources/loki.yml"


def test_loki_and_datasource_configs_parse():
    cfg = yaml.safe_load(LOKI_CFG.read_text())
    assert cfg["auth_enabled"] is False
    assert cfg["common"]["storage"]["filesystem"]["chunks_directory"].startswith("/loki")
    assert cfg["compactor"]["retention_enabled"] is True and cfg["limits_config"]["retention_period"] == "72h"
    assert cfg["schema_config"]["configs"][0]["schema"] == "v13"
    ds = yaml.safe_load(DATASOURCE.read_text())["datasources"][0]
    assert ds["type"] == "loki" and ds["uid"] == "ihds-loki" and ds["url"] == "http://loki:3100"


def test_compose_has_loki_alloy_and_grafana_wiring():
    c = yaml.safe_load(COMPOSE.read_text())
    s = c["services"]
    assert {"loki", "alloy", "grafana", "prometheus", "api"} <= s.keys()
    assert re.fullmatch(r"grafana/loki:\d+\.\d+\.\d+", s["loki"]["image"])  # pinned
    assert re.fullmatch(r"grafana/alloy:v\d+\.\d+\.\d+", s["alloy"]["image"])  # pinned
    assert "127.0.0.1:3100:3100" in s["loki"]["ports"]  # bound to localhost only
    assert "/usr/bin/loki" in s["loki"]["healthcheck"]["test"][1]
    # alloy: config, host log dir, API log volume; starts after Loki is healthy
    vols = " ".join(s["alloy"]["volumes"])
    assert "alloy-config.alloy:/etc/alloy/config.alloy" in vols and "./logs:/var/log/ihds-host" in vols
    assert "ihds-logs:/var/log/ihds-container" in vols
    assert s["alloy"]["depends_on"]["loki"]["condition"] == "service_healthy"
    # api writes the file Alloy tails
    assert s["api"]["environment"]["IHDS_LOG_FILE"] == "/app/logs/ihds.jsonl"
    assert "ihds-logs:/app/logs" in s["api"]["volumes"]
    # grafana mounts the provisioning tree that contains the Loki datasource and the logs dashboard
    assert any(v.startswith("./monitoring/grafana/provisioning:") for v in s["grafana"]["volumes"])
    assert any(v.startswith("./monitoring/grafana/dashboards:") for v in s["grafana"]["volumes"])
    # every named volume used by a service is declared
    used = {v.split(":")[0] for sv in s.values() for v in sv.get("volumes", []) if not v.startswith(".")}
    assert used <= set(c["volumes"]), used - set(c["volumes"])


def test_alloy_config_labels_are_low_cardinality_and_paths_match_compose():
    txt = ALLOY_CFG.read_text()
    code = "\n".join(line for line in txt.splitlines() if not line.strip().startswith("//"))
    labels = re.search(r"stage\.labels\s*\{\s*values\s*=\s*\{(.*?)\}", code, re.S).group(1)
    assert set(re.findall(r"(\w+)\s*=", labels)) == {"level", "route"}  # request_id is NOT a label
    assert "request_id" not in re.search(r"stage\.labels.*?\n  \}", code, re.S).group(0)
    assert 'url = "http://loki:3100/loki/api/v1/push"' in code
    assert '"job" = "ihds-api"' in code
    for mount in ("/var/log/ihds-host", "/var/log/ihds-container"):
        assert mount in code
    assert code.count("{") == code.count("}")


def test_logs_dashboard_logql_uses_labels_alloy_creates():
    d = json.loads(DASH.read_text())
    assert d["uid"] == "ihds-logs" and len(d["panels"]) >= 10
    allowed_labels = {"job", "source", "level", "route"}  # stream labels from alloy-config.alloy
    json_fields = set(lc.ALLOWED_FIELDS)
    for p in d["panels"]:
        assert p["datasource"]["uid"] == "ihds-loki"
        for t in p["targets"]:
            e = t["expr"]
            selector = re.search(r"\{([^}]*)\}", e).group(1)
            used = set(re.findall(r"(\w+)\s*(?:=~|!~|!=|=)", selector))
            assert used <= allowed_labels, (p["title"], used)
            assert 'job="ihds-api"' in selector
            assert "request_id" not in re.sub(r"\|=\s*\"\$request_id\"", "", e)  # only as a text filter, never a label
            # fields read after `| json` must be real log fields
            for f in re.findall(r"\|\s*(\w+)\s*(?:=|!=)\s*\"", e.split("| json", 1)[1] if "| json" in e else ""):
                assert f in json_fields, (p["title"], f)
            if "unwrap" in e:
                assert re.search(r"unwrap (\w+)", e).group(1) in json_fields
    titles = " ".join(p["title"].lower() for p in d["panels"])
    assert "error" in titles and "escalat" in titles and "recent logs" in titles


def test_logs_artifacts_are_ignored_by_git_and_docker():
    assert "logs/" in (ROOT / ".gitignore").read_text().splitlines()
    di = (ROOT / ".dockerignore").read_text().splitlines()
    assert "logs" in di and "monitoring/grafana/dashboards/ihds_logs.json" in di
