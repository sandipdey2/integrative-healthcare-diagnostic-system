"""Evidence retrieval: ChromaDB (local ONNX MiniLM) with the TF-IDF keyword matcher as fallback."""
import importlib.util
import sys

import pytest

from ihds import workflow as wf

HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
GLUCOSE_Q = "glucose OGTT prediabetes impaired tolerance diabetes threshold BMI overweight obesity risk factor"


def test_keyword_mode_is_the_original_matcher():
    kb = wf.KnowledgeBase(method="keyword")
    ev, method = kb.retrieve(GLUCOSE_Q, 3)
    assert method == "keyword" and [e["id"] for e in ev] == [e["id"] for e in kb.keyword_search(GLUCOSE_Q, 3)]
    assert {e["id"] for e in ev} == {"K1", "K2", "K4"}


def test_falls_back_to_keyword_when_chromadb_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)  # import chromadb -> ImportError
    monkeypatch.delitem(sys.modules, "ihds.vector_store", raising=False)
    kb = wf.KnowledgeBase(method="auto")
    ev, method = kb.retrieve(GLUCOSE_Q, 3)
    assert method == "keyword" and ev and "chromadb" in (kb.chroma_error or "")


def test_falls_back_when_a_chroma_query_fails():
    kb = wf.KnowledgeBase(method="auto")

    class Broken:
        embedding = "x"

        def search(self, *a):
            raise RuntimeError("boom")

    kb._chroma = Broken()
    ev, method = kb.retrieve(GLUCOSE_Q, 3)
    assert method == "keyword" and ev and "boom" in kb.chroma_error


@pytest.mark.skipif(not HAS_CHROMA, reason="chromadb not installed")
def test_chroma_retrieval_is_relevant_and_keeps_template_drafts_approved():
    kb = wf.KnowledgeBase(method="chroma")
    ev, method = kb.retrieve(GLUCOSE_Q, 3)
    if method != "chroma":
        pytest.skip(f"chroma could not start here: {kb.chroma_error}")
    assert kb.embedding == "onnx-all-MiniLM-L6-v2"
    assert {e["id"] for e in ev} >= {"K1", "K2"} and all(0 < e["score"] <= 1 for e in ev)
    assert kb.search("Is there a family history component?")[0]["id"] == "K6"
    # an off-vocabulary question still gets evidence (the keyword matcher returns nothing here)
    assert kb.keyword_search("Should this patient be screened again?") == []
    assert len(kb.search("Should this patient be screened again?")) == 3
    # the reviewer's grounding check still passes on template drafts built from chroma evidence
    from ihds.model import Contribution

    top = [Contribution("Glucose", 160.0, 1.2), Contribution("BMI", 34.0, 0.5), Contribution("Age", 45.0, 0.2)]
    draft = wf.template_draft(0.72, "high", top, ev)
    assert wf.review(draft, 0.72, ev)["approved"]


def test_rag_search_tool_and_trace_report_the_method():
    from ihds.pipeline import build_system

    s = build_system(tools_mode="inprocess")
    from ihds import agents
    rec = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 72, "SkinThickness": 30, "Insulin": 120, "BMI": 34,
           "DiabetesPedigreeFunction": 0.5, "Age": 45}
    res = agents.run(s.graph, rec)
    assert res.retrieval in ("chroma", "keyword")
    line = [t for t in res.trace if t.startswith("evidence_agent")][0]
    assert f"[retrieval: {res.retrieval}" in line
    out = s.tools.tools["rag_search"].invoke({"query": GLUCOSE_Q, "k": 3})
    assert out["method"] == res.retrieval
