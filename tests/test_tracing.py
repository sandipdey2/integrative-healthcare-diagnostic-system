"""ihds/tracing.py: optional LangSmith tracing. The live test runs only with LANGSMITH_API_KEY set."""
import os

import pytest

from ihds import tracing


def test_off_without_key():
    env = {}
    assert tracing.configure(env=env) is None and "LANGSMITH_TRACING" not in env


def test_secrets_table_turns_it_on_with_project_ihds():
    env = {}
    cfg = tracing.configure(secrets={"langsmith": {"LANGSMITH_API_KEY": "lsv2_dummy"}}, env=env)
    assert cfg["project"] == "ihds"
    assert env["LANGSMITH_TRACING"] == env["LANGCHAIN_TRACING_V2"] == "true"
    assert env["LANGSMITH_PROJECT"] == env["LANGCHAIN_PROJECT"] == "ihds"
    assert env["LANGSMITH_API_KEY"] == "lsv2_dummy"


def test_explicit_false_is_respected():
    env = {"LANGSMITH_API_KEY": "x", "LANGSMITH_TRACING": "false"}
    assert tracing.configure(env=env) is None


def test_mcp_stdio_child_env_has_no_keys(monkeypatch):
    from ihds.tool_backends import stdio_connection

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    monkeypatch.setenv("IHDS_RETRIEVAL", "keyword")
    env = stdio_connection()["env"]
    assert set(env) <= {"PATH", "PYTHONPATH", "HOME", "IHDS_GAN_STEPS", "IHDS_RETRIEVAL", "IHDS_EMBEDDINGS",
                        "IHDS_CHROMA_DIR"}
    assert env["IHDS_RETRIEVAL"] == "keyword" and "lsv2_dummy" not in env.values()


def test_run_url_is_none_when_off_or_lookup_fails(monkeypatch):
    for k in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(k, raising=False)
    assert tracing.run_url("abc") is None  # tracing off
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_dummy")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracing, "_PROJECT_URLS", {})

    def boom(name):
        raise ConnectionError("offline")

    monkeypatch.setattr(tracing, "_project_url", boom)
    assert tracing.run_url("abc") is None  # never raises into the app


def test_run_url_format_and_project_lookup_cached(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_dummy")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "ihds")
    monkeypatch.setattr(tracing, "_PROJECT_URLS", {"ihds": "https://smith.example/o/T/projects/p/P"})
    assert tracing.run_url("r1") == "https://smith.example/o/T/projects/p/P/r/r1?poll=true"


def test_writer_llm_call_is_a_named_child_run_with_token_usage():
    """Offline: the writer's chat-model call is an "llm" run named writer_llm under the writer_agent node of
    the graph run, and its usage metadata is summed into AdvisoryResult.llm_tokens."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langsmith.run_helpers import tracing_context
    from langchain_core.callbacks import BaseCallbackHandler

    from ihds import agents
    from ihds.pipeline import build_system
    from ihds.workflow import DISCLAIMER

    s = build_system(tools_mode="inprocess")
    rec = {"Pregnancies": 2, "Glucose": 140, "BloodPressure": 72, "SkinThickness": 29, "Insulin": 125, "BMI": 33.0,
           "DiabetesPedigreeFunction": 0.45, "Age": 41}
    base = agents.run(s.graph, rec)
    text = (f"Estimated diabetes risk probability: {base.probability:.0%}. "
            + " ".join(f"{e['fact']} [{e['id']}]" for e in base.evidence) + " " + DISCLAIMER)
    usage = {"input_tokens": 300, "output_tokens": 120, "total_tokens": 420}
    llm = GenericFakeChatModel(messages=iter([AIMessage(content=text, usage_metadata=usage)]))
    with tracing_context(enabled=False):  # offline test: never send these fake runs to LangSmith
        res = agents._run(s.graph_with_llm(llm), rec)
    assert res.draft_source == "llm" and res.review["approved"] and res.llm_tokens == 420

    class Recorder(BaseCallbackHandler):
        def __init__(self):
            self.chains, self.llms = {}, []

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kw):
            self.chains[run_id] = (kw.get("name") or (serialized or {}).get("name"), parent_run_id)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, **kw):
            self.llms.append((kw.get("name"), parent_run_id, tags or []))

    rec_cb = Recorder()
    llm = GenericFakeChatModel(messages=iter([AIMessage(content=text, usage_metadata=usage)]))
    with tracing_context(enabled=False):
        s.graph_with_llm(llm).invoke({"record": rec, "question": "", "trace": [], "tool_calls": [], "llm_usage": []},
                                     config={"callbacks": [rec_cb]})
    assert len(rec_cb.llms) == 1
    name, parent, tags = rec_cb.llms[0]
    assert name == "writer_llm" and "attempt:1" in tags
    assert rec_cb.chains[parent][0] == "writer_agent"           # nested under the writer node ...
    graph_run = rec_cb.chains[parent][1]
    assert rec_cb.chains[graph_run][1] is None                    # ... of the graph run (the root here)


@pytest.mark.skipif(not os.environ.get("LANGSMITH_API_KEY"), reason="LANGSMITH_API_KEY not set (live LangSmith test)")
def test_live_run_is_traced_with_tags_and_metadata():
    import time

    from langsmith import Client

    from ihds import agents
    from ihds.pipeline import build_system

    assert tracing.configure() is not None
    s = build_system(tools_mode="inprocess")
    rec = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 72, "SkinThickness": 30, "Insulin": 120, "BMI": 34,
           "DiabetesPedigreeFunction": 0.5, "Age": 45}
    res = agents.run(s.graph, rec)
    assert res.langsmith_run_id
    tracing.flush()
    c = Client()
    run = None
    for _ in range(20):  # the root run is created at start and patched with tags/metadata at the end
        try:
            run = c.read_run(res.langsmith_run_id)
            if f"band:{res.band}" in (run.tags or []):
                break
        except Exception:
            pass
        time.sleep(2)
    assert run.name == "ihds-assessment"
    assert {f"band:{res.band}", f"status:{res.status}", f"draft:{res.draft_source}"} <= set(run.tags)
    md = run.extra["metadata"]
    assert md["risk_band"] == res.band and md["final_status"] == res.status and md["retrieval"] == res.retrieval


@pytest.mark.skipif(not (os.environ.get("LANGSMITH_API_KEY") and os.environ.get("OPENAI_API_KEY")),
                    reason="needs LANGSMITH_API_KEY and OPENAI_API_KEY (live LangSmith + gpt-4o-mini test)")
def test_live_llm_runs_are_children_of_the_root_with_tokens():
    import time

    from langsmith import Client

    from ihds import agents
    from ihds.pipeline import build_system

    assert tracing.configure() is not None
    s = build_system(tools_mode="inprocess")
    rec = {"Pregnancies": 2, "Glucose": 140, "BloodPressure": 72, "SkinThickness": 29, "Insulin": 125, "BMI": 33.0,
           "DiabetesPedigreeFunction": 0.45, "Age": 41}
    res = agents.run(s.graph_with_llm(agents.default_llm()), rec)
    assert res.langsmith_run_id and res.langsmith_url.endswith(f"/r/{res.langsmith_run_id}?poll=true")
    assert res.llm_tokens > 0
    tracing.flush()
    c = Client()
    llm_runs, root = [], None
    for _ in range(30):
        time.sleep(2)
        runs = list(c.list_runs(project_name=tracing.project(), trace_id=res.langsmith_run_id))
        llm_runs = [r for r in runs if r.run_type == "llm" and r.end_time]
        root = next((r for r in runs if str(r.id) == res.langsmith_run_id), None)
        if root and root.end_time and len(llm_runs) >= 1 + res.revisions and (root.total_tokens or 0) > 0:
            break
    assert root is not None and root.name == "ihds-assessment"
    assert llm_runs and all(r.name == "writer_llm" and (r.total_tokens or 0) > 0 for r in llm_runs)
    assert root.total_tokens >= sum(r.total_tokens for r in llm_runs) > 0
    assert root.extra["metadata"]["llm_total_tokens"] == res.llm_tokens


def test_app_url_and_lookup_error_is_kept_without_the_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    assert tracing.app_url() == "https://smith.langchain.com"
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.api.smith.langchain.com")
    assert tracing.app_url() == "https://eu.smith.langchain.com"
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_dummy_secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(tracing, "_PROJECT_URLS", {})

    def boom(name):
        raise PermissionError("403 for key lsv2_dummy_secret")

    monkeypatch.setattr(tracing, "_project_url", boom)
    assert tracing.run_url("r1") is None
    assert tracing.LAST_URL_ERROR.startswith("PermissionError") and "lsv2_dummy_secret" not in tracing.LAST_URL_ERROR
