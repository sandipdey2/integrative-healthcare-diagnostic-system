"""Optional LangSmith tracing of the LangGraph workflow.

On only when a LangSmith API key is present: LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) in the environment, or
`LANGSMITH_API_KEY` in a [langsmith] table of Streamlit secrets. `configure()` then sets LANGSMITH_TRACING /
LANGCHAIN_TRACING_V2=true (unless explicitly set to false) and the project (default "ihds").

Each `agents.run()` becomes one root run "ihds-assessment" with the LangGraph graph, the gpt-4o-mini calls and
the tool calls nested under it, tagged and annotated with the risk band, final status, draft source and retrieval
method. The key is never passed to the MCP stdio tool server (its env is a short allow-list), and
scripts/start_all.sh unsets it for the MCP HTTP server.
"""
from __future__ import annotations

import os

KEY_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
DEFAULT_PROJECT = "ihds"


def _secret(secrets, name):
    if secrets is None:
        return None
    try:
        return (secrets.get("langsmith") or {}).get(name)
    except Exception:
        return None


def configure(secrets=None, env=None) -> dict | None:
    """Turn tracing on if a key is available; returns {"project", "endpoint"} or None (off)."""
    env = os.environ if env is None else env
    key = next((env.get(k) for k in KEY_VARS if env.get(k)), None) or _secret(secrets, "LANGSMITH_API_KEY")
    if not key:
        return None
    if str(env.get("LANGSMITH_TRACING", env.get("LANGCHAIN_TRACING_V2", "true"))).lower() in ("false", "0", "no"):
        return None
    env.setdefault("LANGSMITH_API_KEY", str(key).strip())
    project = env.get("LANGSMITH_PROJECT") or _secret(secrets, "LANGSMITH_PROJECT") or DEFAULT_PROJECT
    endpoint = env.get("LANGSMITH_ENDPOINT") or _secret(secrets, "LANGSMITH_ENDPOINT")
    env["LANGSMITH_TRACING"] = env["LANGCHAIN_TRACING_V2"] = "true"
    env["LANGSMITH_PROJECT"] = env["LANGCHAIN_PROJECT"] = str(project)
    if endpoint:
        env["LANGSMITH_ENDPOINT"] = str(endpoint)
    if env is os.environ:
        _refresh_langsmith_env()
    return {"project": str(project), "endpoint": endpoint or "https://api.smith.langchain.com"}


def _refresh_langsmith_env() -> None:
    """langsmith caches the environment variables it reads (utils.get_env_var is an lru_cache). If anything
    traced or checked tracing before configure() ran, it would keep seeing tracing off; clear that cache."""
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except Exception:
        pass


def enabled() -> bool:
    return bool(next((os.environ.get(k) for k in KEY_VARS if os.environ.get(k)), None)) and \
        str(os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", ""))).lower() == "true"


def project() -> str:
    return os.environ.get("LANGSMITH_PROJECT", DEFAULT_PROJECT)


_PROJECT_URLS: dict[str, str] = {}


def _project_url(name: str) -> str:
    """UI base URL of a project; one API lookup per process and project (only successes are cached)."""
    if name not in _PROJECT_URLS:
        from langsmith import Client

        c = Client()
        proj = c.read_project(project_name=name)
        _PROJECT_URLS[name] = f"{c._host_url}/o/{proj.tenant_id}/projects/p/{proj.id}"
    return _PROJECT_URLS[name]


LAST_URL_ERROR = ""  # why the latest run_url() lookup failed ("" after a success)


def app_url() -> str:
    """LangSmith UI home for the configured endpoint (no API call): the fallback link."""
    ep = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    return ep.replace("://api.", "://", 1).replace("://eu.api.", "://eu.", 1) if "smith.langchain" in ep else ep


def run_url(run_id: str) -> str | None:
    """Best-effort LangSmith UI link for a run; None when tracing is off or the lookup fails (no network,
    project not created yet, key without read access; the reason is kept in LAST_URL_ERROR, without the key).
    `?poll=true` makes the page wait for a run in flight."""
    global LAST_URL_ERROR
    if not run_id or not enabled():
        return None
    try:
        url = f"{_project_url(project())}/r/{run_id}?poll=true"
        LAST_URL_ERROR = ""
        return url
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        for k in KEY_VARS:
            if os.environ.get(k):
                msg = msg.replace(os.environ[k], "***")
        LAST_URL_ERROR = msg[:200]
        return None


def flush() -> None:
    """Wait until queued runs are sent (for scripts and tests; the app does not need it)."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith.run_trees import get_cached_client

        wait_for_all_tracers()
        get_cached_client().flush()
    except Exception:
        pass
