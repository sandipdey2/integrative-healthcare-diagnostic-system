"""Vocareum test: live gpt-4o-mini calls through the Vocareum OpenAI-compatible proxy.

Skipped unless OPENAI_API_KEY is set. Run only these:
    export OPENAI_API_KEY=<your voc-... key>
    export OPENAI_API_BASE=https://openai.vocareum.com/v1
    pytest -m vocareum -v
"""
import os

import pytest

from ihds import agents, workflow

pytestmark = [
    pytest.mark.vocareum,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set (live Vocareum test)"),
]


@pytest.fixture(scope="module")
def llm():
    model = agents.default_llm()
    assert model is not None
    return model


def test_vocareum_base_url_configured():
    base = agents.llm_base_url()
    assert base is None or base.startswith("https://")


def test_vocareum_ping(llm):
    r = llm.invoke("Reply with exactly the single word: pong")
    assert "pong" in r.content.lower()
    assert r.response_metadata.get("model_name", "").startswith("gpt-4o-mini")


def test_vocareum_full_graph_uses_llm_writer(system, llm):
    """Full live graph run. LLM output varies: a draft can (rarely) fail the grounding check three times and be
    escalated to the template, so up to 3 independent runs are allowed; the released draft must be the LLM's."""
    record = system.test.iloc[0][list(system.test.columns[:8])].to_dict()  # held-out test record #0 (p ~ 0.73)
    graph = system.graph_with_llm(llm)
    for _ in range(3):
        r = agents.run(graph, record, "Which factors drive this score?")
        assert r.status == "ok"
        assert "writer_agent: llm draft (attempt 1)" in r.trace  # the LLM writer was called
        if r.draft_source == "llm":
            break
    assert r.draft_source == "llm", r.trace  # LLM writer released, not the template
    assert r.review["approved"], r.review
    assert workflow.DISCLAIMER in r.draft
    assert not r.escalated


def test_vocareum_unsafe_request_refused_before_llm(system, llm, good_record):
    r = agents.run(system.graph_with_llm(llm), good_record, "What metformin dose should she take?")
    assert r.status == "blocked" and r.tool_calls == []
