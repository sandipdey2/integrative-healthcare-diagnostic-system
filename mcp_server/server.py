"""IHDS MCP server (official `mcp` Python SDK, FastMCP).

Exposes the five domain tools over the Model Context Protocol:
    risk_score, explain_contributions, rag_search, population_context, synthetic_neighbors

Run:
    python -m mcp_server.server                                   # stdio (spawned by the agents by default)
    python -m mcp_server.server --transport streamable-http --port 8765   # HTTP at http://127.0.0.1:8765/mcp
The server trains the same seeded model/GAN as the app at start-up (`ihds.pipeline.build_system`).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from ihds.tools import make_tool_functions  # noqa: E402


def create_server(host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    from ihds.pipeline import build_core

    core = build_core(gan_steps=int(os.environ.get("IHDS_GAN_STEPS", "3000")))
    fns = make_tool_functions(core.pipe, core.kb, core.welch, core.synthetic_pool)
    server = FastMCP("ihds-tools", instructions="Diabetes-risk advisory tools (advisory only, not a diagnosis).",
                     host=host, port=port, log_level="WARNING")
    for fn in fns.values():
        server.add_tool(fn, name=fn.__name__, description=fn.__doc__)
    return server


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default=os.environ.get("IHDS_MCP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("IHDS_MCP_PORT", "8765")))
    args = ap.parse_args(argv)
    create_server(args.host, args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
