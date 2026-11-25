"""Assessment audit log (SQLAlchemy 2 + Postgres, with a local SQLite fallback).

Each completed assessment is stored as one row: the eight model input features (JSONB), risk band and
probability, reviewer status, draft source, revision count, escalation, block flag, retrieval method,
workflow route and optional LangSmith run id. No free-text question or draft is stored.

URL resolution (first match wins): DATABASE_URL in the environment, then the Streamlit secret
`[database] url`. With no URL the log is a local SQLite file at data/audit.db (or IHDS_AUDIT_SQLITE).
The table is created on startup. A failed write is logged (credentials redacted), counted, and never
raised into the assessment path.

Backends: "postgres (neon)" when the host is Neon, "postgres" for any other Postgres URL (e.g. the
compose service), "sqlite (local)" otherwise. The connection string is never written to logs, metrics
or the process environment from a Streamlit secret.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from prometheus_client import REGISTRY, Counter
from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from ihds.config import FEATURES, ROOT

log = logging.getLogger("ihds.audit_log")

DEFAULT_SQLITE = ROOT / "data" / "audit.db"


def _audit_counter() -> Counter:
    """Reuse the collector if Streamlit re-imports this module (a second registration raises ValueError)."""
    existing = REGISTRY._names_to_collectors.get("ihds_audit_writes_total")
    if isinstance(existing, Counter):
        return existing
    return Counter("ihds_audit_writes_total", "Assessment audit-log writes", ["result"])


WRITES = _audit_counter()
WRITES.labels("ok")
WRITES.labels("failed")

_lock = threading.Lock()
_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None
_backend = "sqlite (local)"
_configured = False
_writes_ok = 0
_writes_failed = 0


class Base(DeclarativeBase):
    pass


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True,
    )
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_band: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer_status: Mapped[str] = mapped_column(String(32), nullable=False, default="n/a")
    draft_source: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    revisions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blocked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retrieval_method: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    workflow_route: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    langsmith_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    features: Mapped[dict] = mapped_column(JSONB().with_variant(JSON(), "sqlite"), nullable=False, default=dict)


def redact(text: str | None) -> str:
    """Strip credentials from a URL or exception string so it is safe to log."""
    if not text:
        return ""
    out = str(text)
    try:
        parsed = urlparse(out)
        if parsed.scheme in ("postgres", "postgresql", "postgresql+psycopg", "sqlite") and parsed.netloc:
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            auth = parsed.netloc.split("@", 1)
            # replace user:pass@host with ***@host, even when the string is embedded in a larger message
            if "@" in parsed.netloc:
                creds = auth[0]
                if creds:
                    out = out.replace(creds, "***", 1)
            # also hide a bare user:password@ if urlparse failed to split them (rare)
            _ = host, port
    except Exception:
        pass
    # fallback: generic user:password@ pattern (also catches URLs inside OperationalError text)
    import re
    out = re.sub(r"(postgres(?:ql(?:\+psycopg)?)?://)[^/\s]+@", r"\1***@", out)
    out = re.sub(r"(sqlite:////?)[^\s]+", r"\1<redacted>", out)
    return out


def _sqlalchemy_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _backend_for(url: str) -> str:
    if url.startswith("sqlite:"):
        return "sqlite (local)"
    host = (urlparse(url).hostname or "").lower()
    if "neon.tech" in host or host.endswith(".neon.build"):
        return "postgres (neon)"
    return "postgres"


def _secret_url(secrets: Any) -> str | None:
    if secrets is None:
        return None
    try:
        db = secrets.get("database") if hasattr(secrets, "get") else None
        if db is not None and hasattr(db, "get"):
            val = db.get("url") or db.get("DATABASE_URL")
            return str(val) if val else None
        if isinstance(db, str) and db.strip():
            return db
        val = secrets.get("DATABASE_URL") if hasattr(secrets, "get") else None
        return str(val) if val else None
    except Exception:
        return None


def _resolve_url(url: str | None, secrets: Any, env: dict | None) -> str:
    env = os.environ if env is None else env
    raw = (url or env.get("DATABASE_URL") or _secret_url(secrets) or "").strip()
    if raw:
        return raw
    path = Path(env.get("IHDS_AUDIT_SQLITE") or DEFAULT_SQLITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def _make_engine(raw_url: str) -> Engine:
    sa_url = _sqlalchemy_url(raw_url)
    if sa_url.startswith("sqlite:"):
        return create_engine(sa_url, connect_args={"check_same_thread": False}, pool_pre_ping=True, future=True)
    return create_engine(sa_url, pool_pre_ping=True, pool_size=3, max_overflow=2, future=True,
                         connect_args={"connect_timeout": 10})


def configure(url: str | None = None, secrets: Any = None, env: dict | None = None, *, force: bool = False) -> dict:
    """Create the engine and the assessments table. Idempotent unless force=True. Never raises."""
    global _engine, _Session, _backend, _configured
    with _lock:
        if _configured and not force:
            return status()
        raw = _resolve_url(url, secrets, env)
        backend = _backend_for(raw)
        try:
            engine = _make_engine(raw)
            Base.metadata.create_all(engine)
            if _engine is not None:
                _engine.dispose()
            _engine, _Session, _backend, _configured = engine, sessionmaker(engine, expire_on_commit=False), backend, True
            log.info("audit log: %s (host %s)", backend, urlparse(_sqlalchemy_url(raw)).hostname or "local")
        except Exception as exc:
            log.warning("audit log: %s configure failed (%s); falling back to sqlite", backend, redact(str(exc)))
            try:
                raw = _resolve_url(None, None, {"IHDS_AUDIT_SQLITE": (env or os.environ).get("IHDS_AUDIT_SQLITE", str(DEFAULT_SQLITE))})
                engine = _make_engine(raw)
                Base.metadata.create_all(engine)
                if _engine is not None:
                    _engine.dispose()
                _engine = engine
                _Session = sessionmaker(engine, expire_on_commit=False)
                _backend = "sqlite (local)"
                _configured = True
            except Exception as exc2:  # pragma: no cover - last resort
                log.warning("audit log: sqlite fallback failed (%s)", redact(str(exc2)))
                _configured = False
        return status()


def reset() -> None:
    """Drop the current engine (for tests)."""
    global _engine, _Session, _backend, _configured, _writes_ok, _writes_failed
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _Session = None
        _backend = "sqlite (local)"
        _configured = False
        _writes_ok = 0
        _writes_failed = 0


def _ensure() -> bool:
    if not _configured:
        configure()
    return _configured and _Session is not None


def _features(record: dict | None) -> dict:
    out = {}
    for key in FEATURES:
        val = (record or {}).get(key)
        if val is None or isinstance(val, (int, float, bool)):
            out[key] = val
        elif isinstance(val, str):
            try:
                out[key] = float(val) if "." in val else int(val)
            except ValueError:
                out[key] = None
        else:
            try:
                out[key] = float(val)
            except (TypeError, ValueError):
                out[key] = None
    return out


def _reviewer_status(res) -> str:
    if getattr(res, "status", None) == "blocked":
        return "n/a"
    review = getattr(res, "review", None) or {}
    if review.get("approved") is True:
        return "approved"
    if review.get("approved") is False:
        return "rejected"
    return "n/a"


def _count(ok: bool) -> None:
    global _writes_ok, _writes_failed
    result = "ok" if ok else "failed"
    if ok:
        _writes_ok += 1
    else:
        _writes_failed += 1
    try:
        WRITES.labels(result).inc()
    except Exception:
        pass
    try:
        from ihds import metrics_push
        pusher = getattr(metrics_push, "_PUSHER", None)
        if pusher is not None:
            pusher.registry.inc("ihds_audit_writes_total", {"result": result})
    except Exception:
        pass


def record(res, record: dict | None = None, *, workflow_route: str = "") -> bool:
    """Insert one assessment row. Never raises; returns True on success."""
    try:
        if not _ensure():
            _count(False)
            return False
        run_id = (getattr(res, "langsmith_run_id", None) or "") or None
        row = Assessment(
            risk_probability=getattr(res, "probability", None),
            risk_band=getattr(res, "band", None) or None,
            reviewer_status=_reviewer_status(res),
            draft_source=getattr(res, "draft_source", "") or "",
            revisions=int(getattr(res, "revisions", 0) or 0),
            escalated=bool(getattr(res, "escalated", False)),
            blocked=getattr(res, "status", None) == "blocked",
            retrieval_method=getattr(res, "retrieval", "") or "",
            workflow_route=workflow_route or "",
            langsmith_run_id=run_id,
            features=_features(record),
        )
        with _Session() as session:
            session.add(row)
            session.commit()
        _count(True)
        return True
    except Exception as exc:
        log.warning("audit log write failed: %s", redact(str(exc)))
        _count(False)
        return False


def _row_dict(row: Assessment) -> dict:
    ts = row.created_at
    if ts is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {
        "id": row.id,
        "created_at": ts.isoformat() if ts is not None else None,
        "risk_probability": row.risk_probability,
        "risk_band": row.risk_band,
        "reviewer_status": row.reviewer_status,
        "draft_source": row.draft_source,
        "revisions": row.revisions,
        "escalated": row.escalated,
        "blocked": row.blocked,
        "retrieval_method": row.retrieval_method,
        "workflow_route": row.workflow_route,
        "langsmith_run_id": row.langsmith_run_id,
        "features": dict(row.features or {}),
    }


def recent(limit: int = 20) -> list[dict]:
    """Newest rows first. Never raises."""
    try:
        if not _ensure():
            return []
        limit = max(1, min(int(limit), 100))
        with _Session() as session:
            rows = session.scalars(select(Assessment).order_by(Assessment.created_at.desc(), Assessment.id.desc()).limit(limit)).all()
            return [_row_dict(r) for r in rows]
    except Exception as exc:
        log.warning("audit log recent failed: %s", redact(str(exc)))
        return []


def count() -> int:
    try:
        if not _ensure():
            return 0
        with _Session() as session:
            return int(session.scalar(select(func.count()).select_from(Assessment)) or 0)
    except Exception as exc:
        log.warning("audit log count failed: %s", redact(str(exc)))
        return 0


def status() -> dict:
    return {
        "backend": _backend if _configured else "unconfigured",
        "rows": count() if _configured else 0,
        "writes_ok": _writes_ok,
        "writes_failed": _writes_failed,
    }
