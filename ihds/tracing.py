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
    return {"project": str(project), "endpoint": endpoint or "https://api.smith.langchain.com"}


def enabled() -> bool:
    return bool(next((os.environ.get(k) for k in KEY_VARS if os.environ.get(k)), None)) and \
        str(os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", ""))).lower() == "true"


def project() -> str:
    return os.environ.get("LANGSMITH_PROJECT", DEFAULT_PROJECT)


def run_url(run_id: str) -> str | None:
    """Best-effort LangSmith UI link for a run (one API lookup of the project); None on any error."""
    try:
        from langsmith import Client

        c = Client()
        proj = c.read_project(project_name=project())
        return f"{c._host_url}/o/{proj.tenant_id}/projects/p/{proj.id}/r/{run_id}?poll=true"
    except Exception:
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
