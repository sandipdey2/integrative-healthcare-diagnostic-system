"""Export the architecture diagrams straight from code.

Run: python scripts/export_graph.py
* docs/architecture.mmd/.png - the compiled LangGraph (graph.draw_mermaid()).
* docs/deployment.mmd/.png   - the service view, built from: the FastAPI routes in api/main.py, the tool
  names the MCP server actually lists over MCP (stdio session), and which agent called which tool in a real
  borderline-case run (tool-call log).
PNGs are rendered by the public mermaid.ink service; if unreachable only the .mmd files are written.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.runnables.graph_mermaid import draw_mermaid_png  # noqa: E402

from ihds import agents  # noqa: E402
from ihds.config import FEATURES  # noqa: E402
from ihds.pipeline import build_system  # noqa: E402

docs = ROOT / "docs"
docs.mkdir(exist_ok=True)
S = build_system(gan_steps=200, tools_mode="mcp")
graph = S.graph.get_graph()
(docs / "architecture.mmd").write_text(graph.draw_mermaid())
print("wrote docs/architecture.mmd")

# which agent used which MCP tool: run test records until one passes through every specialist
edges = set()
for _, row in S.test.iterrows():
    rec = {f: (None if row[f] != row[f] else float(row[f])) for f in FEATURES}
    r = agents.run(S.graph, rec)
    edges |= {(c["agent"], c["tool"]) for c in r.tool_calls}
    if any(a == "uncertainty_agent" for a, _ in edges) and len(edges) >= 5:
        break

import api.main as api_main  # noqa: E402

routes = sorted(f"{m} {r.path}" for r in api_main.app.routes if getattr(r, "methods", None)
                and r.path.split("/")[1] not in ("docs", "redoc", "openapi.json") for m in r.methods - {"HEAD"})
tools = S.tools.server_tool_names
lines = ["flowchart LR", '  UI["Streamlit app.py<br/>:8501"]', '  subgraph API["FastAPI api/main.py :8000"]']
lines += [f'    R{i}["{r}"]' for i, r in enumerate(routes)]
lines += ["  end", '  subgraph G["LangGraph StateGraph (ihds/agents.py)"]']
for a in sorted({a for a, _ in edges}):
    lines.append(f'    {a}["{a}"]')
lines += ["  end", f'  subgraph MCP["MCP server mcp_server/server.py ({S.tools.transport} here; streamable HTTP :8765 in Docker)"]']
lines += [f'    T_{t}["{t}"]' for t in tools]
lines += ["  end", '  UI -- "IHDS_API_URL (optional)" --> API', '  UI -. "otherwise in-process" .-> G', "  API --> G"]
lines += [f"  {a} -- MCP --> T_{t}" for a, t in sorted(edges)]
mmd = "\n".join(lines) + "\n"
(docs / "deployment.mmd").write_text(mmd)
print("wrote docs/deployment.mmd; tools listed over MCP:", tools, "routes:", routes)
try:
    (docs / "architecture.png").write_bytes(graph.draw_mermaid_png())
    (docs / "deployment.png").write_bytes(draw_mermaid_png(mmd))
    print("wrote docs/architecture.png, docs/deployment.png")
except Exception as exc:  # offline
    print(f"PNG not rendered ({type(exc).__name__}); paste the .mmd files into https://mermaid.live")
S.tools.close()
