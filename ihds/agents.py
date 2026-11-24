"""Multi-agent advisory workflow built with LangGraph + LangChain.

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
  * evidence_agent    - tools `rag_search` (TF-IDF retrieval, agentic-rag-system method) and
                        `population_context` (Welch t-test results, statistical-analysis-project method).
  * uncertainty_agent - tool `synthetic_neighbors`: nearest GAN-generated rows (generative-mnist method)
                        around the patient and the spread of model probabilities over them.
  * writer_agent      - gpt-4o-mini via langchain-openai when OPENAI_API_KEY is set (or an injected chat
                        model); otherwise a deterministic template. Uses reviewer feedback on revisions.
  * reviewer          - critic: disclaimer, citations, probability, banned phrases, grounding score.
  * finalize          - packages the result; if the revision cap is hit, swaps in the template draft
                        and marks the case `escalated` for mandatory human review.

Honest scope notes: routing is rule-based (the LLM only writes text); tools are in-process LangChain
tools, not an MCP server; all agents share one process and one typed state.
"""
from __future__ import annotations

import operator
import os
from datetime import datetime, timezone
from typing import Annotated, Any, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
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
    population: Optional[list]
    uncertainty: Optional[dict]
    draft: str
    draft_source: str
    feedback: list
    review: dict
    revisions: int
    escalated: bool
    next: str
    trace: Annotated[list, operator.add]
    tool_calls: Annotated[list, operator.add]


def _log(tool: StructuredTool, args: dict, result: Any) -> dict:
    return {"tool": tool.name, "args": {k: (v if isinstance(v, (str, int, float)) else "...") for k, v in args.items()},
            "result_preview": str(result)[:120], "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _call(tool: StructuredTool, **args):
    result = tool.invoke(args)
    return result, _log(tool, args, result)


def build_tools(pipe, kb: wf.KnowledgeBase, welch: pd.DataFrame, synthetic_pool: pd.DataFrame) -> dict:
    """LangChain tools wrapping the project's ML / statistics / generative components."""
    imputer, scaler = pipe.named_steps["impute"], pipe.named_steps["scale"]
    pool_z = scaler.transform(synthetic_pool[FEATURES].to_numpy())
    pool_p = pipe.predict_proba(synthetic_pool[FEATURES])[:, 1]

    def risk_score(record: dict) -> float:
        """Probability of diabetes from the logistic-regression pipeline."""
        return round(float(risk_model.predict_proba(pipe, record)[0]), 4)

    def explain_contributions(record: dict) -> list:
        """Per-feature log-odds contributions, largest first."""
        return [c.__dict__ for c in risk_model.contributions(pipe, record)]

    def rag_search(query: str, k: int = 3) -> list:
        """Retrieve clinical knowledge snippets (TF-IDF cosine similarity)."""
        return kb.search(query, k)

    def population_context(feature: str) -> dict:
        """Welch t-test comparison of a feature between diabetic and non-diabetic patients."""
        row = welch[welch["feature"] == feature].iloc[0]
        return {"feature": feature, "mean_no_diabetes": float(row["mean_no_diabetes"]),
                "mean_diabetes": float(row["mean_diabetes"]), "p_value": float(row["p_value"]),
                "cohens_d": float(row["cohens_d"])}

    def synthetic_neighbors(record: dict, k: int = 25) -> dict:
        """Model probabilities over the k nearest GAN-generated rows to this patient (robustness check)."""
        z = scaler.transform(imputer.transform(pd.DataFrame([record])[FEATURES]))[0]
        idx = np.argsort(((pool_z - z) ** 2).sum(axis=1))[:k]
        p = pool_p[idx]
        return {"k": int(k), "mean": round(float(p.mean()), 3), "min": round(float(p.min()), 3),
                "max": round(float(p.max()), 3), "share_above_0_5": round(float((p >= 0.5).mean()), 3)}

    fns = [risk_score, explain_contributions, rag_search, population_context, synthetic_neighbors]
    return {f.__name__: StructuredTool.from_function(f) for f in fns}


def default_llm():
    """gpt-4o-mini via langchain-openai, only when a key is configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=LLM_MODEL, temperature=0, timeout=30, max_retries=1)


class _C:  # light object so template_draft/llm prompt can use .feature/.value/.contribution
    def __init__(self, d):
        self.__dict__.update(d)


def make_nodes(pipe, kb, welch, synthetic_pool, llm=None) -> dict:
    """Agent node functions (each takes the shared state and returns a partial update)."""
    tools = build_tools(pipe, kb, welch, synthetic_pool)

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

    def _clean_record(state):
        return {f: (float("nan") if state["record"].get(f) is None else float(state["record"][f])) for f in FEATURES}

    def risk_agent(state: AgentState):
        rec = _clean_record(state)
        p, l1 = _call(tools["risk_score"], record=rec)
        contribs, l2 = _call(tools["explain_contributions"], record=rec)
        band = wf.risk_band(p)
        return {"probability": p, "band": band, "borderline": BORDERLINE[0] <= p < BORDERLINE[1],
                "contributions": contribs, "tool_calls": [l1, l2],
                "trace": [f"risk_agent: p={p:.3f} band={band}; top driver={contribs[0]['feature']}"]}

    def evidence_agent(state: AgentState):
        top = [_C(c) for c in state["contributions"]]
        query = wf.build_query(top, state.get("question", ""))
        ev, l1 = _call(tools["rag_search"], query=query, k=3)
        pop, l2 = _call(tools["population_context"], feature=top[0].feature)
        return {"evidence": ev, "population": [pop], "tool_calls": [l1, l2],
                "trace": [f"evidence_agent: {[e['id'] for e in ev]}; population context for {top[0].feature}"]}

    def uncertainty_agent(state: AgentState):
        u, l1 = _call(tools["synthetic_neighbors"], record=_clean_record(state), k=25)
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
        if llm is not None:
            evtxt = "\n".join(f"[{e['id']}] {e['fact']}" for e in ev)
            drivers = "; ".join(f"{c.feature}={c.value:g} ({c.contribution:+.2f} log-odds)" for c in top[:3])
            fb = ("\nReviewer feedback to fix: " + "; ".join(state["feedback"])) if state.get("feedback") else ""
            msgs = [
                SystemMessage(content=(
                    "You write short advisory notes for clinicians about a diabetes risk score. Use ONLY the "
                    "evidence provided and cite it as [K#]. Do not diagnose or recommend medication or doses. "
                    "State the probability exactly as given. End with this sentence verbatim: " + wf.DISCLAIMER)),
                HumanMessage(content=(f"Probability: {p:.0%} ({band} band). Drivers: {drivers}.\n"
                                      + "\n".join(extra) + f"\nQuestion: {state.get('question') or 'none'}\n"
                                      f"Evidence:\n{evtxt}{fb}\nAt most 140 words.")),
            ]
            try:
                draft = llm.invoke(msgs).content.strip()
                return {"draft": draft, "draft_source": "llm",
                        "trace": [f"writer_agent: llm draft (attempt {attempt})"]}
            except Exception as exc:
                note = f"llm error {type(exc).__name__}, template used"
                draft = wf.template_draft(p, band, top, ev, state.get("imputed"))
                return {"draft": _with_extra(draft, extra), "draft_source": "template",
                        "trace": [f"writer_agent: {note} (attempt {attempt})"]}
        draft = wf.template_draft(p, band, top, ev, state.get("imputed"))
        return {"draft": _with_extra(draft, extra), "draft_source": "template",
                "trace": [f"writer_agent: template draft (attempt {attempt})"]}

    def reviewer(state: AgentState):
        rv = wf.review(state["draft"], state["probability"], state["evidence"])
        failed = [k for k, ok in rv["checks"].items() if not ok]
        verdict = "approved" if rv["approved"] else "rejected: " + ", ".join(failed)
        upd = {"review": rv, "feedback": failed, "trace": [f"reviewer: {verdict} (grounding {rv['grounding_score']})"]}
        if not rv["approved"]:
            upd["revisions"] = state.get("revisions", 0) + 1
        return upd

    def finalize(state: AgentState):
        if state["review"]["approved"]:
            return {"status": "ok", "trace": ["finalize: approved draft released for clinician review"]}
        top = [_C(c) for c in state["contributions"]]
        draft = wf.template_draft(state["probability"], state["band"], top, state["evidence"], state.get("imputed"))
        rv = wf.review(draft, state["probability"], state["evidence"])
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


def build_graph(pipe, kb, welch, synthetic_pool, llm=None):
    nodes = make_nodes(pipe, kb, welch, synthetic_pool, llm)
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
    return g.compile()


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


def run(graph, record: dict, question: str = "") -> AdvisoryResult:
    s = graph.invoke({"record": dict(record), "question": question or "", "trace": [], "tool_calls": []},
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
    )
