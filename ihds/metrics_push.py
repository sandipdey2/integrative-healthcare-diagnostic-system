"""Push IHDS metrics straight to a Prometheus remote-write endpoint (Grafana Cloud) from the Streamlit process.

Streamlit Community Cloud cannot run a Prometheus binary or expose /metrics, so this module keeps a tiny
in-process registry and a daemon thread sends it every IHDS_PUSH_INTERVAL seconds (default 15) with the
Prometheus remote-write 1.0 protocol: a protobuf WriteRequest, snappy-compressed, HTTP basic auth.

No extra dependencies: the protobuf message (three nested types) and snappy's block format (literal-only
encoding, which every snappy decoder accepts) are written by hand below; HTTP uses urllib.

Metric names match api/main.py, so monitoring/grafana/dashboards/ihds_api.json works for both sources:
  ihds_assess_total{status,draft_source}  ihds_predictions_total{band}  ihds_escalations_total
  ihds_reviewer_revisions_total           ihds_assess_duration_seconds (histogram)
  ihds_app_up (1 while the process runs)  ihds_app_start_time_seconds
Every series carries service="ihds-streamlit" (IHDS_METRICS_SERVICE), source="streamlit",
deployment=IHDS_METRICS_DEPLOYMENT (default "streamlit-cloud") and instance=<hostname>.

Off unless GRAFANA_RW_URL is configured (env or st.secrets). Pushing never raises into the app: failures are
counted, logged without credentials, and retried on the next tick. The token is kept in this module only; it is
never written to os.environ, so subprocesses (e.g. the MCP stdio tool server) do not inherit it.
"""
from __future__ import annotations

import base64
import logging
import math
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.request

log = logging.getLogger("ihds.metrics_push")

ASSESS_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
CONFIG_KEYS = ("GRAFANA_RW_URL", "GRAFANA_RW_USER", "GRAFANA_RW_TOKEN")


# --------------------------------------------------------------------------- wire format
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field_bytes(num: int, payload: bytes) -> bytes:  # wire type 2 (length-delimited)
    return _varint(num << 3 | 2) + _varint(len(payload)) + payload


def encode_write_request(series: list[tuple[dict[str, str], float, int]]) -> bytes:
    """prometheus.WriteRequest{timeseries=1}; TimeSeries{labels=1, samples=2}; Label{name=1, value=2};
    Sample{value=1 (double), timestamp=2 (int64 ms)}. Labels are sorted by name, as remote write requires."""
    out = bytearray()
    for labels, value, ts_ms in series:
        ts = bytearray()
        for k in sorted(labels):
            ts += _field_bytes(1, _field_bytes(1, k.encode()) + _field_bytes(2, str(labels[k]).encode()))
        samples = value if isinstance(value, list) else [(value, ts_ms)]  # list = several (value, ts) samples
        for v, t in samples:
            ts += _field_bytes(2, b"\x09" + struct.pack("<d", float(v)) + b"\x10" + _varint(int(t)))
        out += _field_bytes(1, bytes(ts))
    return bytes(out)


def snappy_block(data: bytes) -> bytes:
    """Snappy block format using literal elements only (valid snappy; no compression, which is fine for a
    few KB): uncompressed length as varint, then literals of at most 64 KiB, tag 0xF0 + 2-byte length-1."""
    out = bytearray(_varint(len(data)))
    for i in range(0, len(data), 65536):
        chunk = data[i:i + 65536]
        n = len(chunk) - 1
        if n < 60:
            out.append(n << 2)
        else:
            out.append(61 << 2)  # 2 length bytes follow
            out += struct.pack("<H", n)
        out += chunk
    return bytes(out)


# --------------------------------------------------------------------------- registry
class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = {}
        self.new_keys: set[tuple] = set()
        self.hist_buckets: dict[float, int] = {b: 0 for b in ASSESS_BUCKETS}
        self.hist_count = 0
        self.hist_sum = 0.0
        self.start_time = time.time()

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            if key not in self.counters:
                self.new_keys.add(key)  # first push also sends a 0 sample, so increase() counts the first event
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, seconds: float) -> None:
        with self._lock:
            self.hist_count += 1
            self.hist_sum += seconds
            for b in ASSESS_BUCKETS:
                if seconds <= b:
                    self.hist_buckets[b] += 1

    def record_assessment(self, res, seconds: float | None = None) -> None:
        self.inc("ihds_assess_total", {"status": res.status or "unknown", "draft_source": res.draft_source or "none"})
        if res.band:
            self.inc("ihds_predictions_total", {"band": res.band})
        # always present so rate()/increase() work from the first escalation
        self.inc("ihds_escalations_total", value=1.0 if res.escalated else 0.0)
        self.inc("ihds_reviewer_revisions_total", value=float(res.revisions or 0))
        if seconds is not None:
            self.observe(seconds)

    def snapshot(self, base: dict[str, str], ts_ms: int | None = None,
                 zero_ts_ms: int | None = None) -> list[tuple[dict[str, str], float]]:
        """With ts_ms, counters not yet delivered get [(0, zero_ts_ms), (value, ts_ms)]: an explicit 0 one push
        interval earlier, so increase() counts the first event without distorting its extrapolation."""
        with self._lock:
            rows = []
            for (name, lbl), v in self.counters.items():
                fresh = ts_ms is not None and (name, lbl) in self.new_keys
                rows.append(({**base, **dict(lbl), "__name__": name}, [(0.0, zero_ts_ms or ts_ms - 1), (v, ts_ms)] if fresh else v))

            name = "ihds_assess_duration_seconds"
            for b in ASSESS_BUCKETS:
                rows.append(({**base, "__name__": f"{name}_bucket", "le": _fmt_le(b)}, self.hist_buckets[b]))
            rows.append(({**base, "__name__": f"{name}_bucket", "le": "+Inf"}, self.hist_count))
            rows.append(({**base, "__name__": f"{name}_count"}, self.hist_count))
            rows.append(({**base, "__name__": f"{name}_sum"}, self.hist_sum))
        rows.append(({**base, "__name__": "ihds_app_up"}, 1.0))
        rows.append(({**base, "__name__": "ihds_app_start_time_seconds"}, self.start_time))
        return rows


def _fmt_le(b: float) -> str:
    return str(float(b)) if not math.isinf(b) else "+Inf"


# --------------------------------------------------------------------------- pusher
class Pusher:
    def __init__(self, url: str, user: str = "", token: str = "", interval: float = 15.0, timeout: float = 5.0,
                 labels: dict[str, str] | None = None, registry: Registry | None = None) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError("GRAFANA_RW_URL must be an http(s) URL")
        if url.startswith("http://") and user:
            raise ValueError("refusing to send credentials over http://")
        self.url, self.interval, self.timeout = url, interval, timeout
        self._auth = ("Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()) if user else None
        self.registry = registry or Registry()
        self.labels = labels or {}
        self.pushes_ok = 0
        self.pushes_failed = 0
        self.last_status: int | str | None = None
        self._last_ts = 0
        self.include_prometheus_client = False  # also push ihds_* from prometheus_client's REGISTRY (embedded API)
        self._prom_seen: set[tuple] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:  # never show the credentials
        return f"Pusher(url={self.url!r}, ok={self.pushes_ok}, failed={self.pushes_failed})"

    def push_once(self) -> bool:
        ts = max(int(time.time() * 1000), self._last_ts + 1)  # strictly increasing per series
        self._last_ts = ts
        with self.registry._lock:
            fresh = set(self.registry.new_keys)
        zero_ts = ts - max(1, int(self.interval * 1000))
        rows = self.registry.snapshot(self.labels, ts, zero_ts)
        prom_new: set[tuple] = set()
        if self.include_prometheus_client:
            prom_rows, prom_new = self._prometheus_client_rows(ts, zero_ts)
            # one series per label set: the API's REGISTRY wins (e.g. ihds_assess_duration_seconds exists in both;
            # the app records into the API's metrics while the embedded API is in use)
            taken = {tuple(sorted(lbl.items())) for lbl, _ in prom_rows}
            rows = [r for r in rows if tuple(sorted(r[0].items())) not in taken] + prom_rows
        body = snappy_block(encode_write_request([(lbl, v, ts) for lbl, v in rows]))
        headers = {"Content-Type": "application/x-protobuf", "Content-Encoding": "snappy",
                   "X-Prometheus-Remote-Write-Version": "0.1.0", "User-Agent": "ihds-metrics-push/1"}
        if self._auth:
            headers["Authorization"] = self._auth
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url, data=body, headers=headers, method="POST"),
                                        timeout=self.timeout) as r:
                self.last_status = r.status
            with self.registry._lock:
                self.registry.new_keys -= fresh  # zero samples delivered; retried on failure
            self._prom_seen |= prom_new
            self.pushes_ok += 1
            return True
        except urllib.error.HTTPError as e:
            self.last_status = e.code
            detail = e.read(300).decode("utf-8", "replace")
            log.warning("metrics push rejected: HTTP %s %s", e.code, detail)
        except Exception as e:  # network errors, timeouts: never propagate into the app
            self.last_status = type(e).__name__
            log.warning("metrics push failed: %s", type(e).__name__)
        self.pushes_failed += 1
        return False

    def _prometheus_client_rows(self, ts: int, zero_ts: int):
        """ihds_* samples of the in-process FastAPI app (api/main.py registers them in prometheus_client's default
        REGISTRY), relabelled with this pusher's labels. ihds_app_* gauges are skipped (the pusher sends its own
        heartbeat) and so are *_created timestamps. New counter/histogram series get a leading 0 sample."""
        try:
            from prometheus_client import REGISTRY
        except ImportError:
            return [], set()
        rows, new = [], set()
        for fam in REGISTRY.collect():
            if not fam.name.startswith("ihds_") or fam.name.startswith("ihds_app_"):
                continue
            cumulative = fam.type in ("counter", "histogram")
            for smp in fam.samples:
                if smp.name.endswith("_created"):
                    continue
                labels = {**self.labels, **smp.labels, "__name__": smp.name}
                key = (smp.name, tuple(sorted(smp.labels.items())))
                if cumulative and key not in self._prom_seen:
                    new.add(key)
                    rows.append((labels, [(0.0, zero_ts), (smp.value, ts)]))
                else:
                    rows.append((labels, smp.value))
        return rows, new

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.push_once()
            except Exception:  # pragma: no cover - defensive
                log.exception("metrics push loop error")
            self._stop.wait(self.interval)

    def start(self) -> Pusher:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="ihds-metrics-push", daemon=True)
            self._thread.start()
        return self

    def stop(self, final_push: bool = True) -> None:
        self._stop.set()
        if final_push:
            self.push_once()


# --------------------------------------------------------------------------- process-wide singleton
_PUSHER: Pusher | None = None
_INIT_LOCK = threading.Lock()


def config_from(env: dict | None = None, secrets=None) -> dict[str, str]:
    """GRAFANA_RW_* from the environment first, then st.secrets (top level or a [grafana] table)."""
    env = os.environ if env is None else env
    cfg: dict[str, str] = {}
    for k in CONFIG_KEYS:
        v = env.get(k)
        if not v and secrets is not None:
            try:
                v = secrets.get(k) or (secrets.get("grafana") or {}).get(k)
            except Exception:
                v = None
        if v:
            cfg[k] = str(v).strip()
    return cfg


def start(env: dict | None = None, secrets=None) -> Pusher | None:
    """Start the background pusher once per process. Returns None (feature off) without GRAFANA_RW_URL."""
    global _PUSHER
    with _INIT_LOCK:
        if _PUSHER is not None:
            return _PUSHER
        e = os.environ if env is None else env
        cfg = config_from(e, secrets)
        if not cfg.get("GRAFANA_RW_URL"):
            return None
        if cfg.get("GRAFANA_RW_USER") and not cfg.get("GRAFANA_RW_TOKEN"):
            log.warning("metrics push off: GRAFANA_RW_USER set but GRAFANA_RW_TOKEN empty")
            return None
        labels = {"job": "ihds-streamlit", "service": e.get("IHDS_METRICS_SERVICE", "ihds-streamlit"),
                  "source": "streamlit", "deployment": e.get("IHDS_METRICS_DEPLOYMENT", "streamlit-cloud"),
                  "instance": socket.gethostname()}
        try:
            _PUSHER = Pusher(cfg["GRAFANA_RW_URL"], cfg.get("GRAFANA_RW_USER", ""), cfg.get("GRAFANA_RW_TOKEN", ""),
                             interval=float(e.get("IHDS_PUSH_INTERVAL", "15")), labels=labels).start()
        except Exception as exc:
            log.warning("metrics push off: %s", exc)
            return None
        log.warning("metrics push: sending ihds_* series to %s every %ss", cfg["GRAFANA_RW_URL"], _PUSHER.interval)
        return _PUSHER


def record_assessment(res, seconds: float | None = None) -> None:
    """Safe to call whether or not the pusher is running; never raises."""
    try:
        if _PUSHER is not None:
            _PUSHER.registry.record_assessment(res, seconds)
    except Exception:  # pragma: no cover - defensive
        log.exception("metrics record failed")


def include_api_metrics(on: bool = True) -> None:
    """Push the in-process FastAPI app's prometheus_client metrics too (used with ihds/embedded_api.py)."""
    if _PUSHER is not None:
        _PUSHER.include_prometheus_client = on


def status() -> dict | None:
    if _PUSHER is None:
        return None
    return {"url": _PUSHER.url, "ok": _PUSHER.pushes_ok, "failed": _PUSHER.pushes_failed,
            "last_status": _PUSHER.last_status, "api_metrics": _PUSHER.include_prometheus_client}
