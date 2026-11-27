"""Optional Arize AX tracing of the LangGraph workflow (OpenTelemetry + OpenInference).

Independent of LangSmith (ihds/tracing.py): either, both or neither can be on.

On only when ARIZE_SPACE_ID and ARIZE_API_KEY are both set in the environment (and IHDS_ARIZE is not 0/false/off).
`configure()` then calls `arize.otel.register(...)` (project ARIZE_PROJECT_NAME, default "ihds-diagnostic-agents")
and instruments LangChain with `openinference-instrumentation-langchain`, which also covers the LangGraph nodes and
the ChatOpenAI (gpt-4o-mini) calls. Off, it is a no-op: nothing is imported and no span is created.

Each `agents.run()` is wrapped in one parent span "ihds-assessment" (kind AGENT) that carries only coarse result
attributes: ihds.risk_band, ihds.probability (2 decimals), ihds.escalated, ihds.revisions, ihds.draft_source,
ihds.grounding_score, ihds.status, ihds.retrieval, ihds.borderline, ihds.llm_total_tokens. The LangGraph, node and
LLM spans nest under it.

Privacy: by default the instrumentor is configured to HIDE inputs and outputs (span input/output values, LLM
prompts/messages and completions, tool arguments). Patient feature values and the free-text question are never
set as attributes by this module. Set IHDS_ARIZE_CAPTURE_CONTENT=1 to turn the hiding off (not recommended).
Span timing, names, token counts and the model name are still exported.

The Arize keys are never logged and never passed to the MCP tool server (its env is an allow-list).
"""
from __future__ import annotations

import contextlib
import logging
import os

log = logging.getLogger("ihds.arize")

DEFAULT_PROJECT = "ihds-diagnostic-agents"
SPAN_NAME = "ihds-assessment"
_OFF = ("0", "false", "no", "off")

# TraceConfig switches turned on (= hidden) unless IHDS_ARIZE_CAPTURE_CONTENT=1.
HIDE_FLAGS = ("hide_inputs", "hide_outputs", "hide_input_messages", "hide_output_messages", "hide_prompts",
              "hide_choices", "hide_llm_invocation_parameters")

_STATE: dict = {"config": None, "provider": None, "instrumentor": None}


def wanted(env=None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("ARIZE_SPACE_ID") and env.get("ARIZE_API_KEY")
                and str(env.get("IHDS_ARIZE", "1")).strip().lower() not in _OFF)


def enabled() -> bool:
    """True once configure() has switched tracing on in this process."""
    return _STATE["config"] is not None


def capture_content(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("IHDS_ARIZE_CAPTURE_CONTENT", "0")).strip().lower() in ("1", "true", "yes", "on")


def trace_config_kwargs(env=None) -> dict:
    """The TraceConfig keyword arguments: everything hidden unless content capture is explicitly on."""
    hide = not capture_content(env)
    return {k: hide for k in HIDE_FLAGS}


def configure(env=None) -> dict | None:
    """Turn Arize tracing on if configured. Returns {"project", "content_hidden"} (no secrets) or None (off).

    Idempotent (a Streamlit rerun or a second import does not register twice). Never raises: a missing package or
    a failed registration leaves the app running with tracing off."""
    env = os.environ if env is None else env
    if _STATE["config"] is not None:
        return _STATE["config"]
    if not wanted(env):
        return None
    try:
        from arize.otel import register
        from openinference.instrumentation import TraceConfig
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError as e:  # optional packages: the app works without them
        log.warning("Arize tracing requested but packages are missing (%s); tracing stays off", e.name)
        return None
    project = env.get("ARIZE_PROJECT_NAME") or DEFAULT_PROJECT
    try:
        provider = register(space_id=env["ARIZE_SPACE_ID"].strip(), api_key=env["ARIZE_API_KEY"].strip(),
                            project_name=project, verbose=False)
        instrumentor = LangChainInstrumentor()
        instrumentor.instrument(tracer_provider=provider, config=TraceConfig(**trace_config_kwargs(env)))
    except Exception as e:  # tracing must never break the app; log the type only (messages could echo headers)
        log.warning("Arize tracing could not start (%s); tracing stays off", type(e).__name__)
        return None
    _STATE.update(provider=provider, instrumentor=instrumentor,
                  config={"project": project, "content_hidden": not capture_content(env)})
    return _STATE["config"]


def flush(timeout_ms: int = 10000) -> bool:
    """Export any buffered spans now (the exporter batches). True if the provider reported success."""
    provider = _STATE["provider"]
    if provider is None:
        return False
    try:
        return bool(provider.force_flush(timeout_ms))
    except Exception:
        return False


def shutdown() -> None:
    """Undo configure() (used by tests)."""
    inst, provider = _STATE["instrumentor"], _STATE["provider"]
    with contextlib.suppress(Exception):
        if inst is not None:
            inst.uninstrument()
    with contextlib.suppress(Exception):
        if provider is not None:
            provider.shutdown()
    _STATE.update(config=None, provider=None, instrumentor=None)


def result_attributes(res) -> dict:
    """The coarse, non-identifying attributes recorded on the assessment span (no inputs, no free text)."""
    review = res.review or {}
    attrs = {
        "ihds.status": res.status,
        "ihds.risk_band": res.band or "none",
        "ihds.borderline": bool(res.borderline),
        "ihds.escalated": bool(res.escalated),
        "ihds.revisions": int(res.revisions or 0),
        "ihds.draft_source": res.draft_source or "none",
        "ihds.retrieval": res.retrieval or "none",
        "ihds.llm_total_tokens": int(res.llm_tokens or 0),
    }
    if res.probability is not None:
        attrs["ihds.probability"] = round(float(res.probability), 2)
    if review.get("grounding_score") is not None:
        attrs["ihds.grounding_score"] = round(float(review["grounding_score"]), 3)
    return attrs


class _Noop:
    def annotate(self, res) -> None:
        pass


class _Live:
    def __init__(self, span):
        self.span = span

    def annotate(self, res) -> None:
        try:
            for k, v in result_attributes(res).items():
                self.span.set_attribute(k, v)
        except Exception:
            pass


@contextlib.contextmanager
def assessment_span():
    """Parent span around one agents.run(); yields an object with .annotate(result). No-op when tracing is off."""
    if not enabled():
        yield _Noop()
        return
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("ihds", tracer_provider=_STATE["provider"])
        cm = tracer.start_as_current_span(SPAN_NAME, attributes={"openinference.span.kind": "AGENT"})
    except Exception:
        yield _Noop()
        return
    with cm as span:
        yield _Live(span)
