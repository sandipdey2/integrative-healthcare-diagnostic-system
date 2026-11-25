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
