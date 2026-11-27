r"""Multi-agent advisory workflow built with LangGraph + LangChain.

Graph (exported from code to docs/architecture.mmd / .png by scripts/export_graph.py):

    START -> safety_guard --blocked--> refuse -> END
                         \--ok------> supervisor
    supervisor --(conditional)--> risk_agent | evidence_agent | uncertainty_agent | writer_agent
    risk_agent, evidence_agent, uncertainty_agent -> supervisor          (hub-and-spoke loop)
    writer_agent -> reviewer
    reviewer --approved / cap reached--> finalize -> END
             \--rejected and revisions < MAX_REVISIONS--> writer_agent   (revision loop)

Agents (graph nodes) and the LangChain tools they call:
  * safety_guard      - rule-based input guardrail (ranges, prompt injection, dosing/diagnosis requests).
  * supervisor        - deterministic router: picks the next specialist from what is still missing in
                        the shared state; sends borderline cases to the uncertainty agent.
  * risk_agent        - tools `risk_score`, `explain_contributions` (logistic regression,
                        breast-cancer-classification method).
  * evidence_agent    - tools `rag_search` (ChromaDB embeddings, TF-IDF keyword fallback) and
                        `population_context` (Welch t-test results, statistical-analysis-project method).
  * uncertainty_agent - tool `synthetic_neighbors`: nearest GAN-generated rows (generative-mnist method)
                        around the patient and the spread of model probabilities over them.
  * writer_agent      - gpt-4o-mini via langchain-openai when OPENAI_API_KEY is set (or an injected chat
                        model); otherwise a deterministic template. Uses reviewer feedback on revisions.
  * reviewer          - critic: disclaimer, citations, probability, banned phrases, grounding score.
  * finalize          - packages the result; if the revision cap is hit, swaps in the template draft
                        and marks the case `escalated` for mandatory human review.

Honest scope notes: routing is rule-based (the LLM only writes text). The five tools are served by a real MCP
server (mcp_server/server.py, FastMCP) and reached through ihds/tool_backends.py: by default a stdio child
process started without API keys, or streamable HTTP when IHDS_MCP_URL is set. IHDS_TOOLS=inprocess (or "auto"
mode when the server cannot start, labelled "inprocess-fallback") calls the same functions in this process.
All agents share one process and one typed state; only the tool server can run separately.
"""
from __future__ import annotations

import operator
import os
import socket
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import model as risk_model
from . import workflow as wf
from .config import BORDERLINE, FEATURES, LLM_MODEL

MAX_REVISIONS = 2  # reviewer can send a draft back at most twice (writer runs <= 3 times)
RECURSION_LIMIT = 30


class AgentState(TypedDict, total=False):
    record: dict
    question: str
    status: str                       # "running" | "ok" | "blocked"
    guard_reasons: list
    imputed: list
    probability: Optional[float]
    band: Optional[str]
    borderline: bool
    contributions: list
    evidence: Optional[list]
    retrieval: str                    # "chroma" | "keyword" (which retriever answered rag_search)
    population: Optional[list]
    uncertainty: Optional[dict]
    draft: str
    draft_source: str
    feedback: list
    case_facts: list                  # case data lines the writer is given (also grounding sources)
    review: dict
    revisions: int
    escalated: bool
    next: str
    trace: Annotated[list, operator.add]
    tool_calls: Annotated[list, operator.add]
    llm_usage: Annotated[list, operator.add]  # token usage of each writer LLM call


def llm_base_url() -> str | None:
    """OpenAI-compatible endpoint, e.g. Vocareum's https://openai.vocareum.com/v1 (None = api.openai.com)."""
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or None


def default_llm():
    """gpt-4o-mini via langchain-openai, only when a key is configured.

    Honors OPENAI_BASE_URL / OPENAI_API_BASE so a Vocareum (OpenAI-compatible proxy) key works.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=0, timeout=60, max_retries=2, base_url=llm_base_url())


class _C:  # light object so template_draft/llm prompt can use .feature/.value/.contribution
    def __init__(self, d):
        self.__dict__.update(d)


WRITER_SYSTEM_PROMPT = (
    "You write short advisory notes for clinicians about a diabetes risk score. Use only the case data and the "
    "evidence cards you are given, and keep close to their wording; add no other facts or general advice. "
    "Cite an evidence card as [K#] right after a statement taken from it. Case data (probability, drivers, "
    "population context, robustness check) needs no citation; never cite a card for it. Do not diagnose or "
    "recommend medication or doses. State the probability exactly as given. End with this sentence verbatim: "
    + wf.DISCLAIMER)


def case_facts(p: float, band: str, top: list, extra: list[str], imputed: list | None = None) -> list[str]:
    """The case data lines given to the writer; the reviewer counts them as grounding sources too."""
    drivers = "; ".join(f"{c.feature} = {c.value:g} ({'raises' if c.contribution > 0 else 'lowers'} risk, "
                        f"{c.contribution:+.2f} log-odds)" for c in top[:3])
    lines = [f"Estimated diabetes risk probability: {p:.0%} ({band} band).", f"Main model drivers: {drivers}."]
    if imputed:
        lines.append(f"Not measured (imputed with training median): {', '.join(imputed)}.")
    return lines + list(extra)


def make_nodes(tools, llm=None) -> dict:
    """Agent node functions (each takes the shared state and returns a partial update).

    `tools` is a backend from ihds/tool_backends.py (MCP by default); `tools.call(name, **args)` returns
    (result, log entry) and `tools.transport` names the route ("mcp-stdio", "mcp-http", "inprocess...").
    """
    def _call(agent, name, **args):
        result, log = tools.call(name, **args)
        return result, {"agent": agent, **log}

    # ---------------------------------------------------------------- nodes
    def safety_guard(state: AgentState):
        rec = wf.normalize_record(state["record"])
        g = wf.input_guard(rec, state.get("question", ""))
        note = f" (imputed: {', '.join(g.imputed)})" if g.imputed else ""
        return {"record": rec, "guard_reasons": g.reasons, "imputed": g.imputed,
                "status": "running" if g.allowed else "blocked", "revisions": 0, "escalated": False,
                "feedback": [], "evidence": None, "uncertainty": None, "probability": None,
                "trace": [f"safety_guard: {'ok' + note if g.allowed else 'blocked: ' + '; '.join(g.reasons)}"]}

    def refuse(state: AgentState):
        return {"draft": "Request blocked by safety guard: " + "; ".join(state["guard_reasons"]),
                "draft_source": "refusal", "trace": ["refuse: run stopped"]}

    def supervisor(state: AgentState):
        if state.get("probability") is None:
            nxt = "risk_agent"
        elif state.get("evidence") is None:
            nxt = "evidence_agent"
        elif state.get("borderline") and state.get("uncertainty") is None:
            nxt = "uncertainty_agent"
        else:
            nxt = "writer_agent"
        return {"next": nxt, "trace": [f"supervisor: -> {nxt}"]}

    def _clean_record(state):  # JSON-friendly: missing values stay None (imputed by the tools)
        return {f: (None if state["record"].get(f) is None else float(state["record"][f])) for f in FEATURES}

    def risk_agent(state: AgentState):
        rec = _clean_record(state)
        r1, l1 = _call("risk_agent", "risk_score", record=rec)
        r2, l2 = _call("risk_agent", "explain_contributions", record=rec)
        p, contribs = r1["probability"], r2["contributions"]
        band = wf.risk_band(p)
        return {"probability": p, "band": band, "borderline": BORDERLINE[0] <= p < BORDERLINE[1],
                "contributions": contribs, "tool_calls": [l1, l2],
                "trace": [f"risk_agent: [tools via {tools.transport}] p={p:.3f} band={band}; top driver={contribs[0]['feature']}"]}

    def evidence_agent(state: AgentState):
        top = [_C(c) for c in state["contributions"]]
        query = wf.build_query(top, state.get("question", ""))
        r1, l1 = _call("evidence_agent", "rag_search", query=query, k=3)
        ev = r1["evidence"]
        method = r1.get("method", "keyword")
        how = f"{method} ({r1['embedding']})" if r1.get("embedding") else method
        pop, l2 = _call("evidence_agent", "population_context", feature=top[0].feature)
        return {"evidence": ev, "population": [pop], "tool_calls": [l1, l2], "retrieval": method,
                "trace": [f"evidence_agent: [retrieval: {how}] {[e['id'] for e in ev]}; "
                          f"population context for {top[0].feature}"]}

    def uncertainty_agent(state: AgentState):
        u, l1 = _call("uncertainty_agent", "synthetic_neighbors", record=_clean_record(state), k=25)
        return {"uncertainty": u, "tool_calls": [l1],
                "trace": [f"uncertainty_agent: neighbour p range {u['min']}-{u['max']}"]}

    def writer_agent(state: AgentState):
        top = [_C(c) for c in state["contributions"]]
        p, band, ev = state["probability"], state["band"], state["evidence"]
        extra = []
        if state.get("population"):
            pc = state["population"][0]
            extra.append(f"Population context: mean {pc['feature']} is {pc['mean_diabetes']:g} in diabetic vs "
                         f"{pc['mean_no_diabetes']:g} in non-diabetic patients (Welch p={pc['p_value']:.1e}).")
        if state.get("uncertainty"):
            u = state["uncertainty"]
            extra.append(f"Robustness check: model probability over {u['k']} similar synthetic profiles ranges "
                         f"{u['min']:.0%}-{u['max']:.0%}.")
        attempt = state.get("revisions", 0) + 1
        facts = case_facts(p, band, top, extra, state.get("imputed"))
        if llm is not None:
            evtxt = "\n".join(f"[{e['id']}] {e['fact']}" for e in ev)
            fb = ("\nReviewer feedback to fix:\n- " + "\n- ".join(state["feedback"])) if state.get("feedback") else ""
            msgs = [
                SystemMessage(content=WRITER_SYSTEM_PROMPT),
                HumanMessage(content=("Case data:\n" + "\n".join(facts)
                                      + f"\nQuestion: {state.get('question') or 'none'}\n"
                                      f"Evidence cards:\n{evtxt}{fb}\nAt most 140 words.")),
            ]
            # Named child run under the writer node (and the "ihds-assessment" root) when LangSmith is on;
            # with_config merges with the graph's callbacks, so the run nests and carries token usage.
            call = llm.with_config(run_name="writer_llm", tags=["writer", f"attempt:{attempt}"],
                                   metadata={"attempt": attempt}) if hasattr(llm, "with_config") else llm
            try:
                msg = call.invoke(msgs)
                draft = msg.content.strip()
                usage = dict(getattr(msg, "usage_metadata", None) or {})
                return {"draft": draft, "draft_source": "llm", "case_facts": facts,
                        "llm_usage": [{"attempt": attempt, **{k: usage.get(k, 0) for k in
                                                               ("input_tokens", "output_tokens", "total_tokens")}}],
                        "trace": [f"writer_agent: llm draft (attempt {attempt})"]}
            except Exception as exc:
                note = f"llm error {type(exc).__name__}, template used"
                draft = wf.template_draft(p, band, top, ev, state.get("imputed"))
                return {"draft": _with_extra(draft, extra), "draft_source": "template", "case_facts": facts,
                        "trace": [f"writer_agent: {note} (attempt {attempt})"]}
        draft = wf.template_draft(p, band, top, ev, state.get("imputed"))
        return {"draft": _with_extra(draft, extra), "draft_source": "template", "case_facts": facts,
                "trace": [f"writer_agent: template draft (attempt {attempt})"]}

    def reviewer(state: AgentState):
        rv = wf.review(state["draft"], state["probability"], state["evidence"], case_facts=state.get("case_facts"))
        failed = [k for k, ok in rv["checks"].items() if not ok]
        verdict = "approved" if rv["approved"] else "rejected: " + ", ".join(failed)
        upd = {"review": rv, "feedback": rv["feedback"], "trace": [f"reviewer: {verdict} (grounding {rv['grounding_score']})"]}
        if not rv["approved"]:
            upd["revisions"] = state.get("revisions", 0) + 1
        return upd

    def finalize(state: AgentState):
        if state["review"]["approved"]:
            return {"status": "ok", "trace": ["finalize: approved draft released for clinician review"]}
        top = [_C(c) for c in state["contributions"]]
        draft = wf.template_draft(state["probability"], state["band"], top, state["evidence"], state.get("imputed"))
        rv = wf.review(draft, state["probability"], state["evidence"], case_facts=state.get("case_facts"))
        return {"status": "ok", "escalated": True, "draft": draft, "review": rv,
                "draft_source": f"template (llm draft rejected {state['revisions']}x, escalated)",
                "trace": [f"finalize: revision cap ({MAX_REVISIONS}) reached -> template draft, escalated to human review"]}

    return {"safety_guard": safety_guard, "refuse": refuse, "supervisor": supervisor,
            "risk_agent": risk_agent, "evidence_agent": evidence_agent,
            "uncertainty_agent": uncertainty_agent, "writer_agent": writer_agent,
            "reviewer": reviewer, "finalize": finalize}


# -------------------------------------------------------------------- routing (conditional edges)
def route_after_guard(state: AgentState):
    return "refuse" if state["status"] == "blocked" else "supervisor"


def route_from_supervisor(state: AgentState):
    return state["next"]


def route_after_review(state: AgentState):
    """Approved -> finalize; rejected -> writer again while revisions <= MAX_REVISIONS; else finalize."""
    if state["review"]["approved"]:
        return "finalize"
    return "writer_agent" if state["revisions"] <= MAX_REVISIONS else "finalize"


def build_graph(tools, llm=None):
    nodes = make_nodes(tools, llm)
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "safety_guard")
    g.add_conditional_edges("safety_guard", route_after_guard, {"refuse": "refuse", "supervisor": "supervisor"})
    g.add_edge("refuse", END)
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            {"risk_agent": "risk_agent", "evidence_agent": "evidence_agent",
                             "uncertainty_agent": "uncertainty_agent", "writer_agent": "writer_agent"})
    for n in ("risk_agent", "evidence_agent", "uncertainty_agent"):
        g.add_edge(n, "supervisor")
    g.add_edge("writer_agent", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review, {"writer_agent": "writer_agent", "finalize": "finalize"})
    g.add_edge("finalize", END)
    compiled = g.compile()
    compiled.ihds_transport = tools.transport  # reported in AdvisoryResult.transport
    return compiled


def _with_extra(draft: str, extra: list) -> str:
    if not extra:
        return draft
    lines = draft.split("\n")
    return "\n".join(lines[:-1] + extra + lines[-1:])  # keep disclaimer last


# -------------------------------------------------------------------- public API
from dataclasses import dataclass, field  # noqa: E402


@dataclass
class AdvisoryResult:
    status: str
    probability: float | None = None
    band: str | None = None
    borderline: bool = False
    contributions: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    population: list = field(default_factory=list)
    uncertainty: dict | None = None
    draft: str = ""
    draft_source: str = ""
    review: dict = field(default_factory=dict)
    revisions: int = 0
    escalated: bool = False
    guard_reasons: list = field(default_factory=list)
    imputed: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    transport: str = ""
    retrieval: str = ""            # "chroma" | "keyword"
    langsmith_run_id: str = ""     # set when LangSmith tracing is on (ihds/tracing.py)
    langsmith_url: str = ""        # link to that run in the LangSmith UI ("" if it could not be resolved)
    langsmith_url_error: str = ""  # why the link could not be resolved
    llm_tokens: int = 0            # total tokens of the writer's LLM calls (from the responses' usage metadata)
    case_facts: list = field(default_factory=list)  # case data given to the writer (grounding sources)


def run(graph, record: dict, question: str = "") -> AdvisoryResult:
    """Run the graph once. With LangSmith tracing on, the run is one root trace "ihds-assessment"; with Arize
    tracing on (ihds/arize_tracing.py), it is also one parent span of that name. Both are optional and independent."""
    from . import arize_tracing

    with arize_tracing.assessment_span() as span:
        res = _run_langsmith(graph, record, question)
        span.annotate(res)
        return res


def _run_langsmith(graph, record: dict, question: str) -> AdvisoryResult:
    from . import tracing

    if tracing.enabled():
        try:
            return _run_traced(graph, record, question)
        except ImportError:
            pass
    return _run(graph, record, question)


def _run_traced(graph, record: dict, question: str) -> AdvisoryResult:
    from langsmith import traceable

    from . import tracing
    from langsmith.run_helpers import get_current_run_tree

    def _summary(res):
        return {"status": res.status, "band": res.band, "probability": res.probability,
                "draft_source": res.draft_source, "retrieval": res.retrieval, "draft": res.draft,
                "review": res.review, "trace": res.trace}

    @traceable(name="ihds-assessment", run_type="chain", tags=["ihds"], process_outputs=_summary)
    def ihds_assessment(record: dict, question: str = ""):
        res = _run(graph, record, question)
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_tags([f"band:{res.band or 'none'}", f"status:{res.status}", f"draft:{res.draft_source or 'none'}",
                         f"retrieval:{res.retrieval or 'none'}"])
            rt.add_metadata({"risk_band": res.band, "final_status": res.status, "draft_source": res.draft_source,
                             "retrieval": res.retrieval, "escalated": res.escalated, "revisions": res.revisions,
                             "probability": res.probability, "tool_transport": res.transport,
                             "approved": bool(res.review.get("approved")) if res.review else None,
                             "llm_calls": sum(t.startswith("writer_agent: llm draft") for t in res.trace),
                             "llm_total_tokens": res.llm_tokens, "hostname": socket.gethostname()})
            res.langsmith_run_id = str(rt.id)
            res.langsmith_url = tracing.run_url(res.langsmith_run_id) or ""
            res.langsmith_url_error = "" if res.langsmith_url else tracing.LAST_URL_ERROR
        return res

    return ihds_assessment(dict(record), question or "")


def _run(graph, record: dict, question: str = "") -> AdvisoryResult:
    s = graph.invoke({"record": dict(record), "question": question or "", "trace": [], "tool_calls": [], "llm_usage": []},
                     config={"recursion_limit": RECURSION_LIMIT})
    return AdvisoryResult(
        status=s["status"], probability=s.get("probability"), band=s.get("band"),
        borderline=bool(s.get("borderline", False)),
        contributions=[risk_model.Contribution(**c) for c in s.get("contributions", [])],
        evidence=s.get("evidence") or [], population=s.get("population") or [],
        uncertainty=s.get("uncertainty"), draft=s.get("draft", ""), draft_source=s.get("draft_source", ""),
        review=s.get("review", {}), revisions=s.get("revisions", 0), escalated=bool(s.get("escalated")),
        guard_reasons=s.get("guard_reasons", []), imputed=s.get("imputed", []),
        trace=s["trace"], tool_calls=s["tool_calls"],
        transport=(s["tool_calls"][0].get("transport", "") if s["tool_calls"] else getattr(graph, "ihds_transport", "")),
        retrieval=s.get("retrieval") or "", case_facts=s.get("case_facts") or [],
        llm_tokens=int(sum(u.get("total_tokens", 0) for u in s.get("llm_usage") or [])),
    )
