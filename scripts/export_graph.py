"""Export the compiled LangGraph to docs/architecture.mmd (+ .png) straight from code.

Run: python scripts/export_graph.py
The PNG is rendered by LangGraph's draw_mermaid_png() (uses the public mermaid.ink service);
if that is unreachable only the .mmd file is written.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ihds.pipeline import build_system  # noqa: E402

docs = ROOT / "docs"
docs.mkdir(exist_ok=True)
graph = build_system(gan_steps=200).graph.get_graph()
(docs / "architecture.mmd").write_text(graph.draw_mermaid())
print("wrote docs/architecture.mmd")
try:
    (docs / "architecture.png").write_bytes(graph.draw_mermaid_png())
    print("wrote docs/architecture.png")
except Exception as exc:  # offline
    print(f"PNG not rendered ({type(exc).__name__}); paste docs/architecture.mmd into https://mermaid.live")
