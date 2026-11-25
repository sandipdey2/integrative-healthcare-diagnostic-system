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
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from ihds import agents, audit_log, monitoring, tracing, workflow
from ihds.config import FEATURES, LLM_MODEL, REPORTS_DIR

tracing.configure()  # LangSmith tracing only when LANGSMITH_API_KEY is in the environment

STATE: dict = {}
WORKFLOW_ROUTE = "rest-api"  # set to "embedded-api" by ihds/embedded_api.py

# Prometheus metrics (default process-wide registry).
REQUESTS = Counter("ihds_http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("ihds_http_request_duration_seconds", "HTTP request latency", ["method", "route"],
                    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
PREDICTIONS = Counter("ihds_predictions_total", "Completed /assess runs by risk band", ["band"])
ASSESS_STATUS = Counter("ihds_assess_total", "/assess runs by workflow status and draft source",
                        ["status", "draft_source"])
ESCALATIONS = Counter("ihds_escalations_total", "/assess runs escalated to a clinician")
REVISIONS = Counter("ihds_reviewer_revisions_total", "Reviewer-requested LLM redrafts across /assess runs")
ASSESS_LATENCY = Histogram("ihds_assess_duration_seconds", "Agent workflow duration per /assess run",
                           buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
APP_UP = Gauge("ihds_app_up", "1 while this IHDS process is running (heartbeat; also pushed by the Streamlit app)")
APP_START = Gauge("ihds_app_start_time_seconds", "Unix time this IHDS process started")
APP_UP.set(1)
# expose the common series at 0 from the start, so increase() in the dashboard counts the first event
for _status, _src in (("ok", "template"), ("ok", "llm"), ("blocked", "refusal")):
    ASSESS_STATUS.labels(_status, _src)
for _band in ("low", "moderate", "high"):
    PREDICTIONS.labels(_band)
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
