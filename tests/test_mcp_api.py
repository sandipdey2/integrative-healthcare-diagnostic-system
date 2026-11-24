"""MCP server (real subprocess, stdio + streamable HTTP), MCP-backed graph, and the FastAPI service."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from ihds import agents
from ihds.config import FEATURES, ROOT
from ihds.pipeline import UNSAFE_QUESTIONS
from ihds.tool_backends import InProcessToolBackend, MCPToolBackend, make_backend, stdio_connection
from ihds.tools import TOOL_NAMES

GAN_STEPS = 800  # same as the `system` fixture so both sides build identical models


def _rec(system, i):
    row = system.test.iloc[i]
    return {f: (None if row[f] != row[f] else float(row[f])) for f in FEATURES}


def test_mcp_stdio_server_lists_and_calls_tools(system):
    """Raw MCP client (official SDK) against `python -m mcp_server.server` over stdio."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    conn = stdio_connection(GAN_STEPS)
    params = StdioServerParameters(command=conn["command"], args=conn["args"], env=conn["env"], cwd=conn["cwd"])
    rec = _rec(system, 0)

    async def go():
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = sorted(t.name for t in (await s.list_tools()).tools)
            out = await s.call_tool("risk_score", {"record": rec})
            ev = await s.call_tool("rag_search", {"query": "glucose threshold", "k": 2})
            return names, json.loads(out.content[0].text), json.loads(ev.content[0].text)

    names, risk, ev = asyncio.run(go())
    assert names == sorted(TOOL_NAMES)
    expected = float(system.pipe.predict_proba(system.test[FEATURES].iloc[[0]])[:, 1][0])
    assert risk["probability"] == pytest.approx(expected, abs=1e-4)
    assert len(ev["evidence"]) == 2 and all("id" in e for e in ev["evidence"])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server():
    port = _free_port()
    env = {**stdio_connection(GAN_STEPS)["env"]}
    proc = subprocess.Popen([sys.executable, "-m", "mcp_server.server", "--transport", "streamable-http",
                             "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("MCP HTTP server did not start")
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=10)


def test_mcp_http_backend_matches_inprocess(system, http_server):
    mcp = MCPToolBackend(url=http_server)
    local = InProcessToolBackend(system.pipe, system.kb, system.welch, system.synthetic_pool)
    try:
        assert mcp.transport == "mcp-http" and sorted(mcp.tools) == sorted(TOOL_NAMES)
        rec = _rec(system, 3)
        for name, args in [("risk_score", {"record": rec}), ("explain_contributions", {"record": rec}),
                           ("population_context", {"feature": "Glucose"}), ("rag_search", {"query": "BMI", "k": 3}),
                           ("synthetic_neighbors", {"record": rec, "k": 25})]:
            got, log = mcp.call(name, **args)
            want, _ = local.call(name, **args)
            assert got == json.loads(json.dumps(want))  # identical JSON over the wire
            assert log["transport"] == "mcp-http" and log["tool"] == name
    finally:
        mcp.close()


def test_graph_runs_through_mcp_stdio_and_matches_inprocess(system):
    assert system.tools.transport == "mcp-stdio"
    local_graph = agents.build_graph(InProcessToolBackend(system.pipe, system.kb, system.welch, system.synthetic_pool))
    oks = 0
    for i in range(10):
        a = agents.run(system.graph, _rec(system, i))
        b = agents.run(local_graph, _rec(system, i))
        assert a.transport == "mcp-stdio" and b.transport == "inprocess"
        assert a.status == b.status
        if a.status == "blocked":  # e.g. a missing required value: no tools are called
            assert a.tool_calls == []
            continue
        assert any("[tools via mcp-stdio]" in t for t in a.trace)
        assert all(c["transport"] == "mcp-stdio" for c in a.tool_calls)
        assert (a.probability, a.band, a.draft, a.uncertainty) == (b.probability, b.band, b.draft, b.uncertainty)
        oks += 1
    assert oks >= 5


def test_unreachable_mcp_falls_back_only_in_auto_mode(system, monkeypatch):
    monkeypatch.setenv("IHDS_MCP_URL", f"http://127.0.0.1:{_free_port()}/mcp")
    args = (system.pipe, system.kb, system.welch, system.synthetic_pool)
    fb = make_backend(*args, mode="auto")
    assert fb.transport == "inprocess-fallback"
    r = agents.run(agents.build_graph(fb), _rec(system, 0))
    assert any("[tools via inprocess-fallback]" in t for t in r.trace)
    with pytest.raises(Exception):
        make_backend(*args, mode="mcp")


# ------------------------------------------------------------------ FastAPI
@pytest.fixture
def client(system, monkeypatch):
    import api.main as api_main

    monkeypatch.setitem(api_main.STATE, "system", system)
    return TestClient(api_main.app)  # no lifespan: reuse the session's system


def test_api_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["tools_transport"] == "mcp-stdio" and h["tools"] == sorted(TOOL_NAMES)


def test_api_assess_ok_and_blocked(client, system):
    r = client.post("/assess", json={"record": _rec(system, 0)})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and 0 <= d["probability"] <= 1 and d["transport"] == "mcp-stdio"
    assert d["trace"][0].startswith("safety_guard") and d["trace"][-1].startswith("finalize")
    assert "not a diagnosis" in d["draft"].lower()
    blocked = client.post("/assess", json={"record": _rec(system, 0), "question": UNSAFE_QUESTIONS[1]}).json()
    assert blocked["status"] == "blocked" and blocked["tool_calls"] == []
    missing = client.post("/assess", json={"record": {**_rec(system, 0), "Glucose": None}}).json()
    assert missing["status"] == "blocked"


def test_api_validation_and_llm_without_key(client, system, monkeypatch):
    assert client.post("/assess", json={"record": {"Glucose": "high"}}).status_code == 422
    for k in ("OPENAI_API_KEY",):
        monkeypatch.delenv(k, raising=False)
    import api.main as api_main
    api_main.STATE.pop("llm_graph", None)
    assert client.post("/assess", json={"record": _rec(system, 0), "use_llm": True}).status_code == 400


def test_api_monitoring_summary(client):
    d = client.get("/monitoring/summary").json()
    assert {"data_drift", "agent_eval", "llm_vs_template"} <= set(d)


def test_streamlit_uses_api_when_ihds_api_url_set(client, monkeypatch):
    """With IHDS_API_URL set, the app's button calls POST /assess (routed here to the FastAPI TestClient)."""
    import httpx
    from streamlit.testing.v1 import AppTest

    calls = []

    def fake(method):
        def _f(url, **kw):
            calls.append((method, url))
            path = url.replace("http://api.test", "")
            return getattr(client, method)(path, **{k: v for k, v in kw.items() if k == "json"})
        return _f

    monkeypatch.setenv("IHDS_API_URL", "http://api.test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "post", fake("post"))
    monkeypatch.setattr(httpx, "get", fake("get"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert ("post", "http://api.test/assess") in calls
    assert any("Risk probability" in m.label for m in at.metric)
    assert any("mcp-stdio" in str(c.value) for c in at.caption)
