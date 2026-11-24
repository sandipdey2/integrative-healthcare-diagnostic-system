"""How the LangGraph agents reach their tools.

* MCPToolBackend (default): LangChain tools loaded with langchain-mcp-adapters from the real MCP server in
  mcp_server/server.py, either spawned as a stdio subprocess or reached over streamable HTTP (IHDS_MCP_URL).
  One MCP session is kept open for the life of the process on a background asyncio loop.
* InProcessToolBackend: the same functions wrapped as LangChain StructuredTools in this process. Used only
  when explicitly requested (IHDS_TOOLS=inprocess) or, in "auto" mode, when the MCP server cannot start;
  the transport label then says "inprocess-fallback" and it appears in every agent trace.
"""
from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import StructuredTool

from .config import ROOT
from .tools import TOOL_NAMES, make_tool_functions


def _log(name: str, args: dict, result: Any, transport: str) -> dict:
    return {"tool": name, "transport": transport,
            "args": {k: (v if isinstance(v, (str, int, float)) else "...") for k, v in args.items()},
            "result_preview": str(result)[:120], "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class InProcessToolBackend:
    def __init__(self, pipe, kb, welch, pool, label: str = "inprocess"):
        self.transport = label
        self.tools = {n: StructuredTool.from_function(f) for n, f in make_tool_functions(pipe, kb, welch, pool).items()}

    def call(self, name: str, **args):
        result = self.tools[name].invoke(args)
        return result, _log(name, args, result, self.transport)

    def close(self):
        pass


def _parse(result: Any):
    """langchain-mcp-adapters returns the MCP text content (str or list of content blocks)."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        result = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in result)
    return json.loads(result) if isinstance(result, str) else result


def stdio_connection(gan_steps: int | None = None) -> dict:
    """Spawn `python -m mcp_server.server` (stdio). Only PATH/PYTHONPATH/HOME are passed: no API keys."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), "HOME": os.environ.get("HOME", "/tmp")}
    if gan_steps is not None:
        env["IHDS_GAN_STEPS"] = str(gan_steps)
    return {"transport": "stdio", "command": sys.executable, "args": ["-m", "mcp_server.server"],
            "cwd": str(ROOT), "env": env}


class MCPToolBackend:
    def __init__(self, url: str | None = None, gan_steps: int | None = None, timeout: float = 180):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        self.url = url
        self.transport = "mcp-http" if url else "mcp-stdio"
        conn = {"transport": "streamable_http", "url": url} if url else stdio_connection(gan_steps)
        self.client = MultiServerMCPClient({"ihds": conn})
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="ihds-mcp")
        self.thread.start()
        self._ready: concurrent.futures.Future = concurrent.futures.Future()
        self._stop = None
        self._runner = asyncio.run_coroutine_threadsafe(self._run(), self.loop)
        try:
            self._ready.result(timeout=timeout)
        except Exception:
            self.close()
            raise
        atexit.register(self.close)

    async def _run(self):
        from langchain_mcp_adapters.tools import load_mcp_tools

        self._stop = asyncio.Event()
        try:
            async with self.client.session("ihds") as session:
                tools = await load_mcp_tools(session)
                self.tools = {t.name: t for t in tools}
                self.server_tool_names = sorted(self.tools)
                missing = set(TOOL_NAMES) - set(self.tools)
                if missing:
                    raise RuntimeError(f"MCP server is missing tools: {sorted(missing)}")
                self._ready.set_result(True)
                await self._stop.wait()
        except BaseException as exc:  # surface start-up errors to the constructor
            if not self._ready.done():
                self._ready.set_exception(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            else:
                raise

    def call(self, name: str, **args):
        fut = asyncio.run_coroutine_threadsafe(self.tools[name].ainvoke(args), self.loop)
        result = _parse(fut.result(timeout=60))
        return result, _log(name, args, result, self.transport)

    def close(self):
        if self.loop.is_closed():
            return
        if self._stop is not None:
            self.loop.call_soon_threadsafe(self._stop.set)
            try:
                self._runner.result(timeout=10)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)


def make_backend(pipe, kb, welch, pool, mode: str | None = None, gan_steps: int | None = None):
    """mode: 'auto' (default; MCP, falling back to in-process), 'mcp' (MCP or fail), 'inprocess'."""
    mode = (mode or os.environ.get("IHDS_TOOLS", "auto")).lower()
    if mode == "inprocess":
        return InProcessToolBackend(pipe, kb, welch, pool)
    try:
        return MCPToolBackend(url=os.environ.get("IHDS_MCP_URL") or None, gan_steps=gan_steps)
    except Exception as exc:
        if mode == "mcp":
            raise
        print(f"[ihds] MCP server unavailable ({type(exc).__name__}: {exc}); using in-process tools", file=sys.stderr)
        return InProcessToolBackend(pipe, kb, welch, pool, label="inprocess-fallback")
