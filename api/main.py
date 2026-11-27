"""IHDS REST API (FastAPI).

    uvicorn api.main:app --port 8000

POST /assess runs the LangGraph multi-agent workflow (tools via the MCP server by default) and returns the
advisory draft, review, trace and the tool transport used. Each completed assessment is also written to the
audit log (ihds/audit_log.py; Postgres when DATABASE_URL is set, otherwise local SQLite). GET /health,
GET /assessments/recent and GET /monitoring/summary are read-only. Set IHDS_MCP_URL to use a running MCP HTTP server instead of a stdio subprocess.

GET /metrics serves Prometheus metrics (prometheus_client): request count and latency per route, /assess
outcomes by risk band and status, and escalations. Scraped by docker-compose.monitoring.yml (local only);
in the Azure Container Apps deployment only Streamlit's port 8501 is public, so /metrics is not reachable
there. IHDS_METRICS=0 turns the endpoint off (404); deploy/azure/deploy_aci.sh sets it because ACI publishes 8000.

Logs: one JSON object per line (ihds/logging_config.py) to stdout, and to IHDS_LOG_FILE when set. Each request
gets an X-Request-ID; a "request" line (route template, status, latency_ms) is written per call (not for
/metrics) and an "assessment" line (risk band, draft source, escalation, revisions, trace node names) per
/assess. Patient values, the question, LLM text and secrets are never logged. docker-compose.monitoring.yml
ships the file to Loki (local only).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from ihds import agents, arize_tracing, audit_log, logging_config, monitoring, tracing, workflow
from ihds.config import FEATURES, LLM_MODEL, REPORTS_DIR

tracing.configure()  # LangSmith tracing only when LANGSMITH_API_KEY is in the environment
arize_tracing.configure()  # Arize AX tracing only when ARIZE_SPACE_ID and ARIZE_API_KEY are set
LOG = logging_config.configure()  # JSON lines to stdout and, with IHDS_LOG_FILE, to a file that Alloy ships to Loki

STATE: dict = {}
WORKFLOW_ROUTE = "rest-api"  # set to "embedded-api" by ihds/embedded_api.py

# Prometheus metrics (default process-wide registry).


def _metric(cls, name: str, doc: str, labels=(), **kw):
    """Create a metric, or reuse it if this module is imported again in the same process (the Streamlit app
    re-imports project modules after a code change); registering the same name twice would raise."""
    existing = REGISTRY._names_to_collectors.get(name)
    return existing if existing is not None else cls(name, doc, labels, **kw)


_FIRST_IMPORT = "ihds_app_start_time_seconds" not in REGISTRY._names_to_collectors
REQUESTS = _metric(Counter, "ihds_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = _metric(Histogram, "ihds_http_request_duration_seconds", "HTTP request latency", ["method", "route"],
                  buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
PREDICTIONS = _metric(Counter, "ihds_predictions_total", "Completed /assess runs by risk band", ["band"])
ASSESS_STATUS = _metric(Counter, "ihds_assess_total", "/assess runs by workflow status and draft source",
                        ["status", "draft_source"])
ESCALATIONS = _metric(Counter, "ihds_escalations_total", "/assess runs escalated to a clinician")
REVISIONS = _metric(Counter, "ihds_reviewer_revisions_total", "Reviewer-requested LLM redrafts across /assess runs")
ASSESS_LATENCY = _metric(Histogram, "ihds_assess_duration_seconds", "Agent workflow duration per /assess run",
                         buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
APP_UP = _metric(Gauge, "ihds_app_up", "1 while this IHDS process is running (heartbeat; also pushed by the Streamlit app)")
APP_START = _metric(Gauge, "ihds_app_start_time_seconds", "Unix time this IHDS process started")
APP_UP.set(1)
# expose the common series at 0 from the start, so increase() in the dashboard counts the first event
for _status, _src in (("ok", "template"), ("ok", "llm"), ("blocked", "refusal")):
    ASSESS_STATUS.labels(_status, _src)
for _band in ("low", "moderate", "high"):
    PREDICTIONS.labels(_band)
if _FIRST_IMPORT:
    APP_START.set_to_current_time()


class PatientRecord(BaseModel):
    Pregnancies: float | None = None
    Glucose: float | None = None
    BloodPressure: float | None = None
    SkinThickness: float | None = None
    Insulin: float | None = None
    BMI: float | None = None
    DiabetesPedigreeFunction: float | None = None
    Age: float | None = None


class AssessRequest(BaseModel):
    record: PatientRecord
    question: str = ""
    use_llm: bool = Field(False, description=f"Use {LLM_MODEL} for the writer agent (needs OPENAI_API_KEY on the server)")


class AssessResponse(BaseModel):
    status: str
    probability: float | None = None
    band: str | None = None
    borderline: bool = False
    contributions: list[dict] = []
    evidence: list[dict] = []
    population: list[dict] = []
    uncertainty: dict | None = None
    draft: str = ""
    draft_source: str = ""
    review: dict = {}
    revisions: int = 0
    escalated: bool = False
    guard_reasons: list[str] = []
    imputed: list[str] = []
    trace: list[str] = []
    tool_calls: list[dict] = []
    transport: str = ""
    retrieval: str = ""
    langsmith_run_id: str = ""
    langsmith_url: str = ""
    langsmith_url_error: str = ""
    llm_tokens: int = 0
    case_facts: list[str] = []


def get_system():
    if "system" not in STATE:
        from ihds.pipeline import build_system
        STATE["system"] = build_system()
    return STATE["system"]


def get_graph(use_llm: bool):
    S = get_system()
    if not use_llm:
        return S.graph
    if "llm_graph" not in STATE:
        llm = agents.default_llm()
        if llm is None:
            raise HTTPException(status_code=400, detail="use_llm=true but no OPENAI_API_KEY is configured on the server")
        STATE["llm_graph"] = S.graph_with_llm(llm)
    return STATE["llm_graph"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await run_in_threadpool(get_system)
    await run_in_threadpool(audit_log.configure)
    yield
    S = STATE.pop("system", None)
    if S is not None:
        S.tools.close()


app = FastAPI(title="IHDS advisory API", version="1.0",
              description="Advisory diabetes-risk drafts for clinician review. Not a diagnosis.", lifespan=lifespan)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"  # route template, so label cardinality stays bounded
        if path != "/metrics":
            LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
            REQUESTS.labels(request.method, path, str(status)).inc()


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    """Outermost middleware: sets the request id, then writes one "request" log line per call."""
    rid = logging_config.new_request_id(request.headers.get("x-request-id"))
    token = logging_config.request_id_var.set(rid)
    start = time.perf_counter()
    status, exc_type = 500, None
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    except Exception as exc:  # noqa: BLE001 - logged by class name only, then re-raised
        exc_type = type(exc).__name__
        raise
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"  # route template: no ids, no query string
        if path != "/metrics":
            level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
            extra = {"event": "request", "method": request.method, "route": path, "status": status,
                     "latency_ms": round((time.perf_counter() - start) * 1000, 1)}
            if exc_type:
                extra["exc_type"] = exc_type
            LOG.log(level, "request", extra=extra)
        logging_config.request_id_var.reset(token)


def log_assessment(res, req: "AssessRequest", seconds: float) -> None:
    """One "assessment" line: outcome metadata only (no feature values, question, draft or guard-reason text)."""
    LOG.info("assessment", extra={
        "event": "assessment", "assess_status": res.status or "unknown", "band": res.band or "none",
        "draft_source": res.draft_source or "none", "escalated": bool(res.escalated), "revisions": res.revisions,
        "borderline": bool(res.borderline), "blocked": res.status == "blocked",
        "guard_reason_count": len(res.guard_reasons or []), "imputed_count": len(res.imputed or []),
        "evidence_count": len(res.evidence or []), "tool_calls": len(res.tool_calls or []),
        "llm_tokens": res.llm_tokens, "use_llm": req.use_llm,
        "trace_nodes": [t.split(":", 1)[0] for t in res.trace],  # node names only; the text after ":" has numbers
        "latency_ms": round(seconds * 1000, 1)})


def record_assessment(res, seconds: float | None = None) -> None:
    # same names and labels as ihds/metrics_push.py (Streamlit -> Grafana Cloud), so one dashboard covers both
    ASSESS_STATUS.labels(res.status or "unknown", res.draft_source or "none").inc()
    if res.band:
        PREDICTIONS.labels(res.band).inc()
    if res.escalated:
        ESCALATIONS.inc()
    if res.revisions:
        REVISIONS.inc(res.revisions)
    if seconds is not None:
        ASSESS_LATENCY.observe(seconds)


@app.get("/metrics", include_in_schema=False)
def metrics():
    if os.environ.get("IHDS_METRICS", "1") == "0":
        raise HTTPException(status_code=404, detail="metrics disabled (IHDS_METRICS=0)")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    S = get_system()
    return {"status": "ok", "tools_transport": S.tools.transport, "tools": sorted(S.tools.tools),
            "llm_available": workflow.llm_available(), "model": LLM_MODEL}


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest):
    t0 = time.perf_counter()
    res = agents.run(get_graph(req.use_llm), req.record.model_dump(), req.question)
    record_assessment(res, time.perf_counter() - t0)
    log_assessment(res, req, time.perf_counter() - t0)
    audit_log.record(res, req.record.model_dump(), workflow_route=WORKFLOW_ROUTE)
    out = asdict(res)
    out["transport"] = res.transport or get_system().tools.transport
    return out


@app.get("/assessments/recent")
def assessments_recent(limit: int = 20):
    """Newest audit-log rows (metadata + the eight model features; no draft or question text)."""
    limit = max(1, min(int(limit), 100))
    return {"backend": audit_log.status()["backend"], "count": audit_log.count(), "items": audit_log.recent(limit)}


@app.get("/monitoring/summary")
def monitoring_summary():
    llm = Path(REPORTS_DIR) / "llm_eval_summary.json"
    data = {"data_drift": monitoring.load_summary(), "agent_eval": monitoring.load_agent_summary(),
            "llm_vs_template": json.loads(llm.read_text()) if llm.exists() else None}
    if data["data_drift"] is None:
        raise HTTPException(status_code=404, detail="no monitoring reports; run python -m ihds.pipeline")
    return data


@app.get("/schema/features")
def features():
    return {"features": FEATURES}
