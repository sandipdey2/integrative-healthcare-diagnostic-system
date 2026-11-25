"""ihds/embedded_api.py: FastAPI inside the Streamlit process, and its metrics in the Grafana push."""
import http.server
import threading

import httpx
import pytest

from ihds import embedded_api, metrics_push as mp
from tests.test_metrics_push import decode_write_request, snappy_decode

REC = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 72, "SkinThickness": 30, "Insulin": 120, "BMI": 34,
       "DiabetesPedigreeFunction": 0.5, "Age": 45}


def test_gate():
    assert embedded_api.enabled({}) is True
    assert embedded_api.enabled({"IHDS_EMBED_API": "0"}) is False
    assert embedded_api.enabled({"IHDS_API_URL": "http://127.0.0.1:8000"}) is False  # Docker/Azure


class _Recv(http.server.BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):
        type(self).bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200); self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def system():
    from api import main as api_main
    from ihds.pipeline import build_system

    saved = dict(api_main.STATE)
    s = build_system(tools_mode="inprocess")
    yield s
    api_main.STATE.clear(); api_main.STATE.update(saved)


def test_embedded_api_serves_assess_and_its_metrics_are_pushed(system):
    api = embedded_api.EmbeddedAPI(system, timeout=30)
    try:
        assert api.url.startswith("http://127.0.0.1:")
        h = httpx.get(f"{api.url}/health", timeout=10).json()
        assert h["status"] == "ok" and h["tools_transport"] == "inprocess"   # the shared System
        r = httpx.post(f"{api.url}/assess", json={"record": REC}, timeout=60).json()
        assert r["status"] == "ok" and r["retrieval"] in ("chroma", "keyword")
        _Recv.bodies = []
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recv)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        p = mp.Pusher(f"http://127.0.0.1:{srv.server_port}/push", labels={"service": "ihds-streamlit"})
        p.include_prometheus_client = True
        assert p.push_once() and p.push_once()
        srv.shutdown()
        first = decode_write_request(snappy_decode(_Recv.bodies[0]))
        second = decode_write_request(snappy_decode(_Recv.bodies[1]))
        reqs = [(lb, s) for lb, s in first if lb["__name__"] == "ihds_http_requests_total" and lb["route"] == "/assess"]
        assert reqs and all(lb["service"] == "ihds-streamlit" and lb["method"] == "POST" for lb, _ in reqs)
        assert all(len(s) == 2 and s[0][0] == 0 and s[1][0] >= 1 for _, s in reqs)   # zero sample first
        names = {lb["__name__"] for lb, _ in first}
        assert {"ihds_http_request_duration_seconds_bucket", "ihds_assess_total", "ihds_assess_duration_seconds_count",
                "ihds_app_up"} <= names
        assert not any(n.endswith("_created") for n in names)
        assert sum(1 for lb, _ in first if lb["__name__"] == "ihds_app_up") == 1   # only the pusher's heartbeat
        assert all(len(s) == 1 for _, s in second)
        keys = [tuple(sorted(lb.items())) for lb, _ in first]
        assert len(keys) == len(set(keys))   # no duplicate series (both registries have the latency histogram)
    finally:
        api.stop()


def test_streamlit_app_routes_through_the_embedded_api(monkeypatch):
    import sys

    from streamlit.testing.v1 import AppTest

    from ihds.config import ROOT

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("IHDS_API_URL", raising=False)
    sys.path.insert(0, str(ROOT))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    side = " ".join(m.value for m in at.sidebar.markdown) + " ".join(c.value for c in at.sidebar.caption)
    assert "Workflow via REST API http://127.0.0.1:" in side and "embedded in this process" in side
    [b for b in at.button if b.label == "Run advisory workflow"][0].click().run()
    assert not at.exception and any("Risk probability" in m.label for m in at.metric)
    assert any("(embedded)" in c.value and "retrieval:" in c.value for c in at.caption)


def test_langsmith_link_comes_back_through_the_embedded_api(monkeypatch):
    """The graph runs in the uvicorn thread; the run id/url it produced must reach the Streamlit trace panel."""
    import sys

    from streamlit.testing.v1 import AppTest

    from ihds import agents
    from ihds.config import ROOT

    real_run = agents.run

    def traced_run(graph, record, question=""):
        res = real_run(graph, record, question)
        res.langsmith_run_id = "run-123"
        res.langsmith_url = "https://smith.example/o/T/projects/p/P/r/run-123?poll=true"
        return res

    monkeypatch.setattr(agents, "run", traced_run)  # api/main.py calls agents.run at request time
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("IHDS_API_URL", raising=False)
    sys.path.insert(0, str(ROOT))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    [b for b in at.button if b.label == "Run advisory workflow"][0].click().run()
    assert not at.exception
    assert any("(embedded)" in c.value for c in at.caption)
    assert any("[View trace in LangSmith](https://smith.example/o/T/projects/p/P/r/run-123?poll=true)" in m.value
               for m in at.markdown)
