"""ihds/arize_tracing.py: optional Arize AX tracing. Fast and offline: register() is faked or exports to memory."""
import builtins
import logging
import sys
import types

import pytest

from ihds import arize_tracing as at

SECRET_KEY = "ak-test-SECRET-key-123"
SPACE = "U3BhY2U6dGVzdA=="
ON = {"ARIZE_SPACE_ID": SPACE, "ARIZE_API_KEY": SECRET_KEY}


@pytest.fixture(autouse=True)
def _clean_state():
    at.shutdown()
    yield
    at.shutdown()


def _fakes(monkeypatch, calls):
    """Replace the three optional imports with fakes that record how they were called."""
    class FakeProvider:
        def force_flush(self, timeout_ms=0):
            calls["flush"] = timeout_ms
            return True

        def shutdown(self):
            calls["shutdown"] = True

    def register(**kw):
        calls["register"] = kw
        return FakeProvider()

    class TraceConfig:
        def __init__(self, **kw):
            calls["trace_config"] = kw

    class LangChainInstrumentor:
        def instrument(self, **kw):
            calls["instrument"] = kw

        def uninstrument(self):
            calls["uninstrument"] = True

    mods = {"arize": types.ModuleType("arize"), "arize.otel": types.ModuleType("arize.otel"),
            "openinference": types.ModuleType("openinference"),
            "openinference.instrumentation": types.ModuleType("openinference.instrumentation"),
            "openinference.instrumentation.langchain": types.ModuleType("openinference.instrumentation.langchain")}
    mods["arize.otel"].register = register
    mods["openinference.instrumentation"].TraceConfig = TraceConfig
    mods["openinference.instrumentation.langchain"].LangChainInstrumentor = LangChainInstrumentor
    for k, m in mods.items():
        monkeypatch.setitem(sys.modules, k, m)


def test_noop_without_env_vars(monkeypatch):
    calls = {}
    _fakes(monkeypatch, calls)
    for env in ({}, {"ARIZE_SPACE_ID": SPACE}, {"ARIZE_API_KEY": SECRET_KEY}):
        assert at.configure(env=env) is None
    assert calls == {} and not at.enabled()


def test_off_switch(monkeypatch):
    calls = {}
    _fakes(monkeypatch, calls)
    for v in ("0", "false", "off"):
        assert at.configure(env={**ON, "IHDS_ARIZE": v}) is None
    assert calls == {}


def test_noop_assessment_span_when_off():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("result must not be read when tracing is off")

    with at.assessment_span() as span:
        span.annotate(Boom())  # no-op: does not touch the result


def test_register_and_instrumentor_called_when_configured(monkeypatch):
    calls = {}
    _fakes(monkeypatch, calls)
    cfg = at.configure(env=ON)
    assert cfg == {"project": "ihds-diagnostic-agents", "content_hidden": True}
    assert calls["register"]["space_id"] == SPACE and calls["register"]["api_key"] == SECRET_KEY
    assert calls["register"]["project_name"] == "ihds-diagnostic-agents"
    assert "tracer_provider" in calls["instrument"] and "config" in calls["instrument"]
    assert at.enabled() and at.flush() is True and calls["flush"]
    # idempotent: a second call (Streamlit rerun) does not register again
    calls.pop("register")
    assert at.configure(env=ON) == cfg and "register" not in calls
    # the config returned to callers never contains the keys
    assert SECRET_KEY not in repr(cfg) and SPACE not in repr(cfg)


def test_custom_project_name(monkeypatch):
    calls = {}
    _fakes(monkeypatch, calls)
    assert at.configure(env={**ON, "ARIZE_PROJECT_NAME": "my-proj"})["project"] == "my-proj"
    assert calls["register"]["project_name"] == "my-proj"


def test_inputs_and_outputs_hidden_by_default_and_opt_in_shows(monkeypatch):
    calls = {}
    _fakes(monkeypatch, calls)
    assert at.configure(env=ON)["content_hidden"] is True
    assert set(calls["trace_config"]) == set(at.HIDE_FLAGS)
    assert all(v is True for v in calls["trace_config"].values())
    assert {"hide_inputs", "hide_outputs", "hide_input_messages", "hide_output_messages"} <= set(at.HIDE_FLAGS)
    at.shutdown()
    assert at.configure(env={**ON, "IHDS_ARIZE_CAPTURE_CONTENT": "1"})["content_hidden"] is False
    assert all(v is False for v in calls["trace_config"].values())


def test_missing_packages_degrade_gracefully(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name.split(".")[0] in ("arize", "openinference"):
            raise ImportError(f"No module named {name!r}", name=name.split(".")[0])
        return real_import(name, *a, **kw)

    for k in [k for k in sys.modules if k.split(".")[0] in ("arize", "openinference")]:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.WARNING, logger="ihds.arize"):
        assert at.configure(env=ON) is None
    assert not at.enabled()
    assert SECRET_KEY not in caplog.text and SPACE not in caplog.text


def test_register_failure_leaves_tracing_off_and_logs_no_secret(monkeypatch, caplog):
    calls = {}
    _fakes(monkeypatch, calls)

    def boom(**kw):
        raise RuntimeError(f"auth failed for {kw['api_key']}")

    monkeypatch.setattr(sys.modules["arize.otel"], "register", boom)
    with caplog.at_level(logging.DEBUG, logger="ihds.arize"):
        assert at.configure(env=ON) is None
    assert not at.enabled()
    assert SECRET_KEY not in caplog.text and SPACE not in caplog.text


def test_configure_logs_nothing_with_secrets(monkeypatch, caplog):
    calls = {}
    _fakes(monkeypatch, calls)
    with caplog.at_level(logging.DEBUG):
        at.configure(env=ON)
    assert SECRET_KEY not in caplog.text and SPACE not in caplog.text


def test_result_attributes_are_coarse_and_contain_no_free_text():
    from ihds.agents import AdvisoryResult

    res = AdvisoryResult(status="approved", probability=0.7349, band="high", borderline=False, escalated=False,
                         revisions=1, draft_source="llm", retrieval="chroma", llm_tokens=321,
                         review={"grounding_score": 0.61234, "approved": True},
                         draft="Glucose 187 is high [X]", case_facts=["Glucose: 187"], evidence=[{"fact": "f"}])
    a = at.result_attributes(res)
    assert a["ihds.probability"] == 0.73 and a["ihds.grounding_score"] == 0.612
    assert a["ihds.risk_band"] == "high" and a["ihds.draft_source"] == "llm" and a["ihds.revisions"] == 1
    assert a["ihds.escalated"] is False and a["ihds.llm_total_tokens"] == 321
    assert all(k.startswith("ihds.") for k in a)
    assert all(isinstance(v, (str, int, float, bool)) for v in a.values())
    assert "Glucose" not in repr(a) and "187" not in repr(a)


def test_blocked_case_attributes_have_no_probability():
    from ihds.agents import AdvisoryResult

    a = at.result_attributes(AdvisoryResult(status="refused"))
    assert "ihds.probability" not in a and a["ihds.risk_band"] == "none"


def test_mcp_stdio_child_env_has_no_arize_keys(monkeypatch):
    from ihds.tool_backends import stdio_connection

    monkeypatch.setenv("ARIZE_API_KEY", SECRET_KEY)
    monkeypatch.setenv("ARIZE_SPACE_ID", SPACE)
    monkeypatch.setenv("IHDS_RETRIEVAL", "keyword")
    env = stdio_connection()["env"]
    assert not any(k.startswith("ARIZE") for k in env) and SECRET_KEY not in env.values()


def test_real_instrumentation_exports_spans_without_patient_values_or_question(monkeypatch):
    """With the real OpenInference instrumentor and an in-memory exporter (no network): one parent span
    "ihds-assessment" with the coarse ihds.* attributes, LangGraph spans under it, and no patient value, no
    question text and no draft text in any span attribute."""
    pytest.importorskip("openinference.instrumentation.langchain")
    pytest.importorskip("arize.otel")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import arize.otel
    from openinference.instrumentation import TracerProvider

    from ihds import agents
    from ihds.pipeline import build_system

    exporter = InMemorySpanExporter()

    def fake_register(**kw):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider

    monkeypatch.setattr(arize.otel, "register", fake_register)
    assert at.configure(env=ON) is not None
    try:
        s = build_system(tools_mode="inprocess")
        rec = {"Pregnancies": 2, "Glucose": 187, "BloodPressure": 72, "SkinThickness": 29, "Insulin": 125, "BMI": 33.6,
               "DiabetesPedigreeFunction": 0.451, "Age": 41}
        question = "SENTINEL-QUESTION-xyzzy why is my sugar so high"
        res = agents.run(s.graph, rec, question)
        spans = exporter.get_finished_spans()
        parents = [sp for sp in spans if sp.name == at.SPAN_NAME]
        assert len(parents) == 1 and len(spans) > 3
        p = parents[0]
        assert p.attributes["ihds.risk_band"] == res.band
        assert p.attributes["ihds.probability"] == round(res.probability, 2)
        assert p.attributes["openinference.span.kind"] == "AGENT"
        others = [sp for sp in spans if sp is not p]
        assert any(sp.parent is not None and sp.parent.span_id == p.context.span_id for sp in others)
        blob = repr([dict(sp.attributes) for sp in spans])
        assert "SENTINEL-QUESTION" not in blob and "xyzzy" not in blob
        assert "187" not in blob and "33.6" not in blob and "0.451" not in blob
        assert res.draft[:40] not in blob
        for sp in spans:  # values are replaced by the instrumentor's placeholder, not exported
            for k in ("input.value", "output.value"):
                assert sp.attributes.get(k, "__REDACTED__") == "__REDACTED__"
        # an LLM call: prompt and completion are hidden too (the writer's ChatOpenAI call goes through the same path)
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        exporter.clear()
        GenericFakeChatModel(messages=iter([AIMessage(content="COMPLETION-sentinel-qwerty")])).invoke("PROMPT-sentinel-asdf")
        llm_blob = repr([dict(sp.attributes) for sp in exporter.get_finished_spans()])
        assert exporter.get_finished_spans() and "PROMPT-sentinel" not in llm_blob
        assert "COMPLETION-sentinel" not in llm_blob and "llm.input_messages" not in llm_blob
    finally:
        at.shutdown()
