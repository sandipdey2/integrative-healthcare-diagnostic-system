"""Structured JSON logging for the IHDS API (one JSON object per line), shipped to Loki by Grafana Alloy.

Design rules, checked by tests/test_logging.py:

* Only the fields in ALLOWED_FIELDS are ever written. A log call cannot add patient feature values, the
  question, an LLM prompt or draft, or a secret by passing it in ``extra=``: unknown extras are dropped.
* The message text is chosen by the code (constant event names such as "request" and "assessment"); it never
  interpolates request data.
* Exceptions are logged by class name only (``exc_type``). The message and traceback can embed the values
  that caused the error, so they are not written.
* ``request_id`` is a field in the line, not a Loki label (unbounded cardinality); Alloy turns only
  ``level`` and ``route`` into labels.

Where the lines go: stdout always (disable with IHDS_LOG_JSON=0), plus a rotating file when IHDS_LOG_FILE is
set (for example ``logs/ihds.jsonl``; the docker-compose.monitoring.yml shipper tails that file).
"""
from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOGGER_NAME = "ihds.api"

# Every key that may appear in a line besides the five standard ones. Nothing else is emitted.
ALLOWED_FIELDS = (
    "event", "request_id", "method", "route", "status", "latency_ms",
    "band", "draft_source", "assess_status", "escalated", "revisions", "borderline", "blocked", "guard_reason_count",
    "imputed_count", "evidence_count", "tool_calls", "llm_tokens", "use_llm", "trace_nodes", "exc_type",
)

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("ihds_request_id", default="")
_REQUEST_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id(incoming: str | None = None) -> str:
    """Keep a caller-supplied X-Request-ID only if it is short and made of safe characters (no log injection)."""
    if incoming and _REQUEST_ID_OK.match(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid:
            out["request_id"] = rid
        for key in ALLOWED_FIELDS:
            if key != "request_id" and key in record.__dict__:
                out[key] = record.__dict__[key]
        if record.exc_info and record.exc_info[0] is not None:
            out["exc_type"] = record.exc_info[0].__name__
        return json.dumps(out, separators=(",", ":"), default=str)


def configure() -> logging.Logger:
    """Attach the JSON handlers to the ``ihds.api`` logger. Safe to call more than once."""
    log = logging.getLogger(LOGGER_NAME)
    if getattr(log, "_ihds_configured", False):
        return log
    log.setLevel(os.environ.get("IHDS_LOG_LEVEL", "INFO").upper())
    log.propagate = False  # do not duplicate lines through the root logger
    fmt = JsonFormatter()
    if os.environ.get("IHDS_LOG_JSON", "1") != "0":
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        log.addHandler(h)
    path = os.environ.get("IHDS_LOG_FILE", "").strip()
    if path:
        try:
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(Path(path).expanduser(), maxBytes=10_000_000, backupCount=3,
                                                      encoding="utf-8")
            fh.setFormatter(fmt)
            log.addHandler(fh)
        except OSError as exc:  # a read-only or missing volume must not stop the API
            print(f"ihds logging: cannot write {path}: {type(exc).__name__}", file=sys.stderr)
    log._ihds_configured = True  # type: ignore[attr-defined]
    return log
