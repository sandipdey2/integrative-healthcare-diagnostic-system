"""ihds/metrics_push.py: Streamlit -> Prometheus remote write (Grafana Cloud). Offline tests use a local HTTP
receiver; the live test runs only when GRAFANA_RW_URL / GRAFANA_RW_USER / GRAFANA_RW_TOKEN are in the env."""
import base64
import http.server
import os
import struct
import threading
import types

import pytest

from ihds import metrics_push as mp


def _uvarint(b: bytes, i: int):
    shift = n = 0
    while True:
        c = b[i]; i += 1
        n |= (c & 0x7F) << shift
        if not c & 0x80:
            return n, i
        shift += 7


def snappy_decode(b: bytes) -> bytes:
    """Reference decoder for literal + copy elements (enough to check our literal-only encoder)."""
    n, i = _uvarint(b, 0)
    out = bytearray()
    while i < len(b):
        tag = b[i]; i += 1
        if tag & 3 == 0:
            ln = tag >> 2
            if ln >= 60:
                k = ln - 59
                ln = int.from_bytes(b[i:i + k], "little"); i += k
            ln += 1
            out += b[i:i + ln]; i += ln
        else:
            raise AssertionError("unexpected copy element")
    assert len(out) == n
    return bytes(out)


def decode_write_request(b: bytes):
    """Tiny protobuf reader for WriteRequest -> [(labels, [(value, ts)])]."""
    def fields(buf):
        i = 0
        while i < len(buf):
            key, i = _uvarint(buf, i)
            num, wt = key >> 3, key & 7
            if wt == 2:
                ln, i = _uvarint(buf, i); yield num, buf[i:i + ln]; i += ln
            elif wt == 1:
                yield num, buf[i:i + 8]; i += 8
            elif wt == 0:
                v, i = _uvarint(buf, i); yield num, v
    out = []
    for _, ts in fields(b):
        labels, samples = {}, []
        for num, sub in fields(ts):
            if num == 1:
                kv = dict(fields(sub)); labels[kv[1].decode()] = kv[2].decode()
            else:
                kv = dict(fields(sub)); samples.append((struct.unpack("<d", kv[1])[0], kv[2]))
        out.append((labels, samples))
    return out


RES = types.SimpleNamespace(status="ok", draft_source="template", band="high", escalated=False, revisions=2)
BLOCKED = types.SimpleNamespace(status="blocked", draft_source="refusal", band=None, escalated=False, revisions=0)


def test_wire_format_roundtrips():
    body = mp.encode_write_request([({"__name__": "ihds_assess_total", "status": "ok", "b": "ü"}, 3.0, 1700000000123)])
    assert decode_write_request(body) == [({"b": "ü", "__name__": "ihds_assess_total", "status": "ok"},
                                           [(3.0, 1700000000123)])]
    for data in (b"", b"x", b"y" * 59, b"z" * 60, os.urandom(200_000)):
        assert snappy_decode(mp.snappy_block(data)) == data
    try:  # cross-check against a real snappy implementation when one is installed
        import cramjam
        assert bytes(cramjam.snappy.decompress_raw(mp.snappy_block(body * 999))) == body * 999
    except ImportError:
        pass


def test_off_without_url(monkeypatch):
    monkeypatch.setattr(mp, "_PUSHER", None)
    assert mp.start(env={}) is None
    assert mp.start(env={"GRAFANA_RW_URL": "https://x/api/prom/push", "GRAFANA_RW_USER": "1"}) is None  # no token
    mp.record_assessment(RES, 0.1)  # no-op, must not raise
    assert mp.status() is None
    secrets = {"grafana": {"GRAFANA_RW_URL": "https://x", "GRAFANA_RW_USER": "u", "GRAFANA_RW_TOKEN": "t"}}
    assert mp.config_from({}, secrets)["GRAFANA_RW_USER"] == "u"   # [grafana] table in st.secrets
    with pytest.raises(ValueError):
        mp.Pusher("http://example.com/push", user="u", token="t")      # no credentials over plain http


class _Receiver(http.server.BaseHTTPRequestHandler):
    calls: list = []
    code = 200

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls.append((dict(self.headers), body))
        self.send_response(type(self).code); self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def receiver():
    _Receiver.calls, _Receiver.code = [], 200
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_push_sends_remote_write_with_dashboard_metric_names(receiver):
    url = f"https://127.0.0.1:{receiver.server_port}/api/prom/push".replace("https", "http")
    p = mp.Pusher(url, labels={"service": "ihds-streamlit", "source": "streamlit", "instance": "t"})
    p._auth = "Basic " + base64.b64encode(b"123:secret").decode()  # as the https path would set
    p.registry.record_assessment(RES, 0.3)
    p.registry.record_assessment(RES, 0.03)
    p.registry.record_assessment(BLOCKED, 0.01)
    assert p.push_once() and p.pushes_ok == 1
    headers, body = _Receiver.calls[0]
    assert headers["Content-Encoding"] == "snappy" and headers["Content-Type"] == "application/x-protobuf"
    assert headers["X-Prometheus-Remote-Write-Version"] == "0.1.0"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"123:secret").decode()
    got = {(lb["__name__"], tuple(sorted((k, v) for k, v in lb.items() if k not in
            ("__name__", "service", "source", "instance")))): s[0][0] for lb, s in decode_write_request(snappy_decode(body))}
    assert all(lb["service"] == "ihds-streamlit" for lb, _ in decode_write_request(snappy_decode(body)))
    assert got[("ihds_assess_total", (("draft_source", "template"), ("status", "ok")))] == 2
    assert got[("ihds_assess_total", (("draft_source", "refusal"), ("status", "blocked")))] == 1
    assert got[("ihds_predictions_total", (("band", "high"),))] == 2
    assert got[("ihds_escalations_total", ())] == 0
    assert got[("ihds_reviewer_revisions_total", ())] == 4
    assert got[("ihds_assess_duration_seconds_count", ())] == 3
    assert got[("ihds_assess_duration_seconds_bucket", (("le", "0.05"),))] == 2
    assert got[("ihds_assess_duration_seconds_bucket", (("le", "+Inf"),))] == 3
    assert got[("ihds_app_up", ())] == 1
    assert "secret" not in repr(p)


def test_push_failures_never_raise(receiver):
    _Receiver.code = 500
    p = mp.Pusher(f"http://127.0.0.1:{receiver.server_port}/push", timeout=2)
    assert p.push_once() is False and p.last_status == 500
    dead = mp.Pusher("http://127.0.0.1:9/push", timeout=1)   # nothing listens on port 9
    assert dead.push_once() is False and dead.pushes_failed == 1
    dead.start(); dead.stop(final_push=False)


LIVE = all(os.environ.get(k) for k in mp.CONFIG_KEYS)


@pytest.mark.skipif(not LIVE, reason="GRAFANA_RW_URL / GRAFANA_RW_USER / GRAFANA_RW_TOKEN not set (live Grafana Cloud test)")
def test_live_grafana_cloud_accepts_push():
    p = mp.Pusher(os.environ["GRAFANA_RW_URL"], os.environ["GRAFANA_RW_USER"], os.environ["GRAFANA_RW_TOKEN"],
                  timeout=15, labels={"job": "ihds-pytest", "service": "ihds-pytest", "source": "pytest"})
    p.registry.record_assessment(RES, 0.2)
    assert p.push_once(), f"Grafana Cloud rejected the push: HTTP {p.last_status}"
