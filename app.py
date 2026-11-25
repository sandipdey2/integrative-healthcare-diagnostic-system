"""Streamlit app: Integrative Healthcare Diagnostic Support (Project 7).

Run locally:  streamlit run app.py --server.headless true
Optional secret: OPENAI_API_KEY (Streamlit secrets or environment). Without it the
explanation is produced by a deterministic template and everything else still works.
If IHDS_API_URL is set (e.g. http://127.0.0.1:8000) the advisory workflow is called through the
FastAPI service (api/main.py); otherwise the LangGraph graph runs in this process, with its tools
served by the MCP server (mcp_server/server.py, stdio subprocess or IHDS_MCP_URL).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_PACKAGES = ("ihds", "api", "mcp_server")


def _code_version() -> str:
    """Hash of the project's Python sources (ihds/, api/, mcp_server/)."""
    h = hashlib.sha1()
    root = Path(__file__).resolve().parent
    for pkg in _PROJECT_PACKAGES:
        for p in sorted((root / pkg).glob("*.py")):
            h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def _live() -> dict:
    """Process-wide holder that survives Streamlit's hot reload (its source never changes): the code version and
    the System / embedded API / metrics pusher built from it, so the old ones can be shut down."""
    return {}


CODE_VERSION = _code_version()
if _live().get("version") not in (None, CODE_VERSION):
    # Streamlit Cloud applies a git push by rerunning this script with only the changed modules reloaded. Unchanged
    # modules would keep references to the old ones, so drop every project module (except the metrics pusher,
    # whose thread keeps running) and import them again; the cached System and API are rebuilt below.
    for _name in [m for m in sys.modules if m.split(".")[0] in _PROJECT_PACKAGES and m != "ihds.metrics_push"]:
        del sys.modules[_name]
_live()["version"] = CODE_VERSION

from ihds import (  # noqa: E402
    agents, audit_log, data, embedded_api, metrics_push, model, monitoring, stats_analysis, tracing, workflow,
)
from ihds.config import FEATURES, LLM_MODEL, REPORTS_DIR, ROOT, TARGET, VALID_RANGES  # noqa: E402
from ihds.pipeline import build_system  # noqa: E402

st.set_page_config(page_title="Healthcare AI Diagnostic Support", page_icon="🩺", layout="wide")


def _load_secret_key() -> None:
    """Copy OPENAI_API_KEY / OPENAI_API_BASE from st.secrets into the environment (values never displayed)."""
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        if os.environ.get(name):
            continue
        try:
            val = st.secrets.get(name)
        except Exception:  # no secrets file
            val = None
        if val:
            os.environ[name] = str(val)


_load_secret_key()


def _start_metrics_push():
    """Optional push of ihds_* metrics to Grafana Cloud (off unless GRAFANA_RW_URL is in st.secrets or the env).
    The values stay inside ihds.metrics_push: they are not copied into os.environ or shown anywhere."""
    try:
        try:
            secrets = st.secrets if len(st.secrets) else None
        except Exception:  # no secrets file
            secrets = None
        return metrics_push.start(secrets=secrets)
    except Exception:
        return None


_pusher = _start_metrics_push()
if _live().get("pusher") not in (None, _pusher):  # ihds.metrics_push itself was reloaded: one pusher only
    _live()["pusher"].stop()
_live()["pusher"] = _pusher


def _start_tracing():
    """Optional LangSmith tracing (on only with LANGSMITH_API_KEY in the env or a [langsmith] secrets table)."""
    try:
        try:
            secrets = st.secrets if len(st.secrets) else None
        except Exception:
            secrets = None
        return tracing.configure(secrets=secrets)
    except Exception:
        return None


TRACING = _start_tracing()


def _start_audit_log():
    """Optional Postgres audit log (Neon or DATABASE_URL); otherwise a local SQLite file."""
    try:
        try:
            secrets = st.secrets if len(st.secrets) else None
        except Exception:
            secrets = None
        return audit_log.configure(secrets=secrets)
    except Exception:
        return audit_log.status()


_start_audit_log()


@st.cache_data(ttl=3600, show_spinner=False)
def _langsmith_url(run_id: str):
    return tracing.run_url(run_id)


def _langsmith_link(res) -> None:
    """'View trace in LangSmith' under the agent trace. The URL comes from the process that ran the graph
    (in-process, or the REST API, embedded or external); this process resolves it only as a fallback."""
    url = res.langsmith_url or (_langsmith_url(res.langsmith_run_id) if res.langsmith_run_id else None)
    if url:
        st.markdown(f"[View trace in LangSmith]({url}) · run `{res.langsmith_run_id}`")
    elif res.langsmith_run_id:
        st.caption(f"LangSmith run `{res.langsmith_run_id}` in project {tracing.project()} (link unavailable)")
    elif TRACING:
        st.caption("LangSmith tracing is on here, but the workflow returned no run id (tracing off where it ran)")


EXTERNAL_API = os.environ.get("IHDS_API_URL", "").rstrip("/")


@st.cache_resource(show_spinner="Training risk model and GAN (first load or new code only)...", max_entries=1)
def get_system(code_version: str):
    # With an external API (Docker/Azure) the graph runs there; the local system only feeds the other tabs.
    # Otherwise this System (MCP stdio tools) is shared with the embedded API below. Keyed by the code version,
    # so a pushed change rebuilds it; the previous one is closed once nothing serves it any more.
    s = build_system(tools_mode="inprocess" if EXTERNAL_API else None)
    old = _live().get("system")
    if old is not None:
        _live()["retired"] = old
    _live()["system"] = s
    return s


def run_via_api(record: dict, question: str, use_llm: bool) -> agents.AdvisoryResult:
    import httpx

    r = httpx.post(f"{API_URL}/assess", json={"record": record, "question": question, "use_llm": use_llm},
                   timeout=180)
    r.raise_for_status()
    d = r.json()
    d["contributions"] = [model.Contribution(**c) for c in d.get("contributions", [])]
    return agents.AdvisoryResult(**d)


S = get_system(CODE_VERSION)


@st.cache_resource(show_spinner="Starting the embedded REST API (first load or new code only)...", max_entries=1)
def get_embedded_api(code_version: str):
    """Once per process and code version: FastAPI on 127.0.0.1:<free port> in a daemon thread
    (ihds/embedded_api.py). A previous API (older code) is stopped first."""
    old = _live().pop("api", None)
    if old is not None:
        old.stop()
    try:
        api = embedded_api.EmbeddedAPI(S)
        _live()["api"] = api
        return {"url": api.url, "error": None, "startup_s": api.startup_s, "code_version": code_version}
    except Exception as exc:
        return {"url": "", "error": f"{type(exc).__name__}: {exc}"[:200], "startup_s": None,
                "code_version": code_version}


EMBEDDED = get_embedded_api(CODE_VERSION) if (not EXTERNAL_API and embedded_api.enabled()) else None
API_URL = EXTERNAL_API or (EMBEDDED["url"] if EMBEDDED else "")
if EMBEDDED and EMBEDDED["url"]:
    metrics_push.include_api_metrics(True)  # every run: also after the pusher was recreated by a reload
_retired = _live().pop("retired", None)
if _retired is not None:  # its API is stopped, so close its MCP stdio child
    try:
        _retired.tools.close()
    except Exception:
        pass


@st.cache_resource(show_spinner=False, max_entries=1)
def get_llm_graph(code_version: str):
    return S.graph_with_llm(agents.default_llm())

st.title("Healthcare AI Diagnostic Support — Diabetes Risk (advisory only)")
st.caption(
    "Project 7 — Integrative Industry Synthesis. Public Pima Indians Diabetes Database (768 records). "
    "Outputs are advisory drafts for clinician review, not diagnoses."
)

with st.sidebar:
    st.header("Run mode")
    api_health = None
    if API_URL:
        try:
            import httpx
            api_health = httpx.get(f"{API_URL}/health", timeout=10).json()
        except Exception as exc:  # shown below
            api_health = {"error": str(exc)}
    has_key = bool(api_health.get("llm_available")) if api_health else workflow.llm_available()
    use_llm = st.toggle(f"Use {LLM_MODEL} for the explanation draft", value=has_key, disabled=not has_key,
                        help="Needs OPENAI_API_KEY. Without it a deterministic template is used.")
    if has_key and API_URL:
        st.write("LLM key configured on the API server")
    elif has_key:
        base = agents.llm_base_url()
        st.write("LLM key detected" + (f" · endpoint: {base}" if base else " · endpoint: api.openai.com"))
    else:
        st.write("No OPENAI_API_KEY: template mode")
    if API_URL:
        st.write(f"Workflow via REST API {API_URL} · tools: {api_health.get('tools_transport', api_health.get('error'))}")
        if EMBEDDED:
            st.caption(f"REST API embedded in this process (started in {EMBEDDED['startup_s']} s)")
    else:
        st.write(f"Workflow in-process · agent tools via {S.tools.transport}")
        if EMBEDDED and EMBEDDED["error"]:
            st.caption(f"Embedded REST API unavailable, using in-process: {EMBEDDED['error']}")
    if not EXTERNAL_API:
        push = metrics_push.status()
        if push:
            host = push["url"].split("/")[2]
            st.caption(f"Metrics push: on → {host} · {push['ok']} ok / {push['failed']} failed"
                       f" (last {push['last_status']})" + (" · incl. API metrics" if push.get("api_metrics") else ""))
        else:
            st.caption("Metrics push: off (no GRAFANA_RW_URL secret)")
    st.caption(f"LangSmith tracing: on (project {TRACING['project']})" if TRACING
               else "LangSmith tracing: off (no LANGSMITH_API_KEY)")
    try:
        info = audit_log.status()
        st.caption(f"Audit log: {info['backend']} · {info['rows']} rows")
    except Exception:
        st.caption("Audit log: unavailable")
    st.divider()
    st.markdown("**Advisory system only** — clinician review required.")

tabs = st.tabs(["Risk assessment", "Model & data", "Synthetic stress test", "Monitoring (Evidently)", "Architecture", "About & limits", "Audit log"])

# ------------------------------------------------------------------ Risk assessment
with tabs[0]:
    st.subheader("1. Enter or load a patient record")
    test_view = S.test.copy()
    idx = st.selectbox("Load a held-out test record (optional)", ["(manual entry)"] + list(range(len(test_view))))
    if idx == "(manual entry)":
        defaults = {"Pregnancies": 2, "Glucose": 140, "BloodPressure": 72, "SkinThickness": 29, "Insulin": 125,
                    "BMI": 33.0, "DiabetesPedigreeFunction": 0.45, "Age": 41}
        true_label = None
    else:
        row = test_view.iloc[int(idx)]
        defaults = {f: (None if pd.isna(row[f]) else row[f]) for f in FEATURES}
        true_label = int(row[TARGET])

    cols = st.columns(4)
    record = {}
    for i, f in enumerate(FEATURES):
        lo, hi = VALID_RANGES[f]
        optional = f in workflow.OPTIONAL_FEATURES
        with cols[i % 4]:
            if optional:
                measured = st.checkbox(f"{f} measured", value=defaults[f] is not None, key=f"m_{f}_{idx}")
                if not measured:
                    record[f] = None
                    st.caption(f"{f}: not measured (will be imputed)")
                    continue
            default = defaults[f] if defaults[f] is not None else (lo + hi) / 2
            step = 0.01 if f == "DiabetesPedigreeFunction" else (0.1 if f == "BMI" else 1.0)
            record[f] = st.number_input(f, value=float(default), step=step, key=f"{f}_{idx}")
    question = st.text_input("Optional clinician question", placeholder="e.g. Which factors drive this score?")

    if st.button("Run advisory workflow", type="primary"):
        res = None
        if API_URL:
            try:
                res = run_via_api(record, question, bool(use_llm))  # the API counts it in its own metrics
            except Exception as exc:
                if EXTERNAL_API:
                    raise
                st.warning(f"Embedded REST API call failed ({type(exc).__name__}); ran the workflow in-process.")
        if res is None:
            graph = get_llm_graph(CODE_VERSION) if (use_llm and has_key) else S.graph
            t0 = time.perf_counter()
            res = agents.run(graph, record, question)
            if API_URL:  # embedded API fallback: count it in the API's metrics, which the pusher already sends
                from api.main import record_assessment as api_record
                api_record(res, time.perf_counter() - t0)
            else:
                metrics_push.record_assessment(res, time.perf_counter() - t0)  # no-op when the push is off
            audit_log.record(res, record, workflow_route="in-process")
        if res.status == "blocked":
            st.error(res.draft)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Risk probability", f"{res.probability:.0%}")
            c2.metric("Risk band", res.band + (" (borderline)" if res.borderline else ""))
            c3.metric("Reviewer", "approved" if res.review["approved"] else "rejected",
                      f"grounding {res.review['grounding_score']}")
            if res.escalated:
                st.error("Revision cap reached: LLM drafts kept failing review; template draft shown and case escalated.")
            if true_label is not None:
                st.caption(f"Recorded outcome for this test record: {'diabetic' if true_label else 'not diabetic'}")
            st.subheader("2. Advisory draft")
            st.caption(f"Draft source: {res.draft_source} · reviewer revisions: {res.revisions}"
                       + (f" · LLM tokens: {res.llm_tokens}" if res.llm_tokens else ""))
            st.text(res.draft)
            st.warning(workflow.DISCLAIMER)
            with st.expander("Feature contributions (log-odds vs. training average)"):
                st.dataframe(pd.DataFrame([c.__dict__ for c in res.contributions]).round(3), hide_index=True)
            with st.expander(f"Retrieved evidence (evidence_agent, retrieval: {res.retrieval or 'n/a'})"):
                st.dataframe(pd.DataFrame(res.evidence), hide_index=True)
                if res.population:
                    st.json(res.population[0])
            if res.uncertainty:
                with st.expander("Robustness check on GAN neighbours (uncertainty_agent)"):
                    st.json(res.uncertainty)
            with st.expander("Reviewer checks"):
                st.json(res.review)
            with st.expander("Agent trace (LangGraph node order) and tool-call log (audit)"):
                st.caption(f"Workflow route: {'REST API ' + API_URL if API_URL else 'in-process LangGraph'}"
                           f"{' (embedded)' if API_URL and not EXTERNAL_API else ''} · "
                           f"tool transport: {res.transport} · retrieval: {res.retrieval or 'n/a'}")
                _langsmith_link(res)
                st.write(res.trace)
                st.dataframe(pd.DataFrame(res.tool_calls), hide_index=True)

# ------------------------------------------------------------------ Model & data
with tabs[1]:
    raw = data.load_raw()
    m = model.evaluate(S.pipe, S.test)
    st.subheader("Held-out test performance (logistic regression, threshold 0.5)")
    c = st.columns(5)
    for col, k in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(k.upper() if k == "roc_auc" else k.capitalize(), m[k])
    cm = m["confusion_matrix"]
    st.table(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                          index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]))
    st.subheader("Coefficients (standardized features)")
    st.dataframe(model.coefficients(S.pipe).round(3), hide_index=True)
    st.subheader("Welch t-tests: diabetic vs non-diabetic (Project 2 statistical-analysis-project method)")
    st.dataframe(stats_analysis.welch_tests(data.clean(raw)), hide_index=True)
    st.subheader("Impossible-zero audit (treated as missing)")
    st.dataframe(data.missing_audit(raw), hide_index=True)

# ------------------------------------------------------------------ Synthetic
with tabs[2]:
    st.subheader("Tabular GAN (NumPy re-implementation of the Project 5 generative-mnist recipe)")
    hist = pd.DataFrame(S.gan.history)
    hist.index = hist.index * 50
    st.line_chart(hist, x_label="training step", y_label="loss")
    imputed = pd.DataFrame(S.pipe.named_steps["impute"].transform(S.train[FEATURES]), columns=FEATURES)
    st.dataframe(pd.DataFrame({"real_mean": imputed.mean(), "synthetic_mean": S.synthetic.mean(),
                               "real_std": imputed.std(), "synthetic_std": S.synthetic.std()}).round(2))
    p = S.pipe.predict_proba(S.synthetic[FEATURES])[:, 1]
    syn = S.synthetic.assign(probability=p.round(3))
    border = syn[(syn.probability >= 0.4) & (syn.probability < 0.6)]
    st.write(f"{len(border)} of {len(syn)} synthetic rows fall in the borderline band (0.40–0.60); "
             "these are used as stress cases for the advisory workflow.")
    st.dataframe(border.head(20), hide_index=True)

# ------------------------------------------------------------------ Monitoring
with tabs[3]:
    st.subheader("Evidently AI reports")
    summary = monitoring.load_summary()
    if st.button("Regenerate reports now"):
        with st.spinner("Running Evidently..."):
            summary = monitoring.run_all(S.pipe, S.train, S.test, S.synthetic)
    if summary is None:
        st.info("No reports yet. Click 'Regenerate reports now' or run `python -m ihds.pipeline`.")
    else:
        rows = []
        for name in ("holdout", "shifted", "synthetic"):
            r = summary.get(name)
            if r:
                rows.append({"report": name, "drifted features": ", ".join(r["drifted_feature_names"]) or "none",
                             "drift share": r["drifted_columns_share"], "accuracy": r.get("accuracy"),
                             "precision": r.get("precision"), "recall": r.get("recall"), "f1": r.get("f1score")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(summary["notes"]["shifted"] + " " + summary["notes"]["synthetic"])
        agent_sum = monitoring.load_agent_summary()
        if agent_sum:
            st.subheader("Agent output evaluation (Evidently)")
            t = agent_sum["agent_text_eval"]; g = agent_sum["guard_eval"]; d = agent_sum["agent_output_drift"]
            st.dataframe(pd.DataFrame([
                {"check": "drafts with [K#] citation", "value": t["has_citation_rate"]},
                {"check": "drafts with disclaimer", "value": t["has_disclaimer_rate"]},
                {"check": "drafts free of forbidden phrases", "value": t["no_forbidden_phrases_rate"]},
                {"check": "reviewer approved", "value": t["reviewer_approved_rate"]},
                {"check": "mean grounding score", "value": t["mean_grounding"]},
                {"check": "guard accuracy on labelled cases", "value": g["accuracy"]},
                {"check": "guard recall (unsafe blocked)", "value": g["recall"]},
            ]), hide_index=True)
            st.caption(f"Guard misses: {g['missed'] or 'none'}. Agent outputs drifted under the simulated glucose "
                       f"shift: {', '.join(d['drifted_outputs']) or 'none'} (mean probability "
                       f"{d['mean_probability_holdout']} -> {d['mean_probability_shifted']}).")
        for name in ("holdout", "shifted", "synthetic", "agent_text_eval", "agent_output_drift", "guard_eval"):
            path = REPORTS_DIR / f"{name}_report.html"
            if path.exists():
                st.download_button(f"Download {name} HTML report", path.read_bytes(), file_name=path.name,
                                   mime="text/html", key=f"dl_{name}")
        pick = st.selectbox("Show a report inline", ["(none)", "holdout", "shifted", "synthetic",
                                                     "agent_text_eval", "agent_output_drift", "guard_eval"])
        if pick != "(none)" and (REPORTS_DIR / f"{pick}_report.html").exists():
            st.components.v1.html((REPORTS_DIR / f"{pick}_report.html").read_text(), height=900, scrolling=True)

# ------------------------------------------------------------------ Architecture
with tabs[4]:
    img = ROOT / "docs" / "architecture.png"
    if img.exists():
        st.image(str(img))
    dep = ROOT / "docs" / "deployment.png"
    if dep.exists():
        st.image(str(dep), caption="Service view (exported from code): Streamlit → FastAPI → LangGraph → MCP tool server")
    st.markdown((ROOT / "diagrams" / "system_architecture.md").read_text()
                .replace("![LangGraph agent graph](../docs/architecture.png)", "")
                .replace("![service view exported from code](../docs/deployment.png)", ""))

# ------------------------------------------------------------------ About
with tabs[5]:
    st.markdown(
        """
- **Data:** Pima Indians Diabetes Database (Smith et al., 1988) via OpenML #37 — 768 females aged 21+ of Pima heritage.
  Results may not transfer to other populations.
- **What is real code:** a LangGraph `StateGraph` with 7 agent nodes (safety guard, supervisor, risk, evidence,
  uncertainty, writer, reviewer) plus refuse/finalize nodes; LangChain tools wrapping the logistic-regression model,
  Welch t-test results, TF-IDF retrieval and the NumPy GAN; a reviewer→writer revision loop capped at 2 revisions;
  Evidently reports on data and on agent outputs.
- **What is not built:** the supervisor routes by rules (the LLM only writes the draft); no MCP server;
  no EHR integration; no user accounts.
- **Human-in-the-loop:** every output is an advisory draft; the clinician decides.
- **Known gaps found in evaluation:** keyword guard missed a paraphrased medication question; the GAN does
  not match the Insulin / SkinThickness distributions well (imputation spikes); a +25 mg/dL glucose
  calibration shift lowers precision. See TESTING.md and evaluation/results.json.
"""
    )


# ------------------------------------------------------------------ Audit log
with tabs[6]:
    st.subheader("Assessment audit log")
    st.caption(
        "One row per completed assessment: the eight model input features, risk band, reviewer status, "
        "draft source, revisions, escalation, retrieval method and workflow route. The clinician question "
        "and the draft text are not stored."
    )
    info = audit_log.status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Backend", info["backend"])
    c2.metric("Rows", info["rows"])
    c3.metric("Write failures", info["writes_failed"])
    rows = audit_log.recent(20)
    if not rows:
        st.info("No assessments logged yet in this process.")
    else:
        view = []
        for r in rows:
            feat = r.get("features") or {}
            view.append({
                "id": r["id"],
                "created_at": r["created_at"],
                "band": r["risk_band"],
                "probability": r["risk_probability"],
                "reviewer": r["reviewer_status"],
                "draft": r["draft_source"],
                "revisions": r["revisions"],
                "escalated": r["escalated"],
                "blocked": r["blocked"],
                "retrieval": r["retrieval_method"],
                "route": r["workflow_route"],
                "langsmith_run_id": r["langsmith_run_id"] or "",
                **{k: feat.get(k) for k in ("Glucose", "BMI", "Age")},
            })
        st.dataframe(pd.DataFrame(view), hide_index=True)
        with st.expander("Raw rows (last 20)"):
            st.json(rows)
