"""Compare evidence retrieval: TF-IDF keyword matcher vs ChromaDB (local ONNX MiniLM, and OpenAI embeddings when
OPENAI_API_KEY is set), and optionally the live gpt-4o-mini reviewer approval rate with each retriever.

    python scripts/compare_retrieval.py                # top-3 per method for the workflow's queries + questions
    python scripts/compare_retrieval.py --llm 30       # + live LLM eval on 30 held-out cases per retriever

Writes reports/retrieval_comparison.json (no credentials).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ihds import workflow as wf  # noqa: E402

DRIVER_PAIRS = [("Glucose", "BMI"), ("Glucose", "Age"), ("BMI", "DiabetesPedigreeFunction"),
                ("Pregnancies", "Glucose"), ("Age", "BloodPressure"), ("Insulin", "SkinThickness")]
QUESTIONS = ["Which factors drive this score?", "Is there a family history component?",
             "What does an oral glucose tolerance result of 160 mean?", "Should this patient be screened again?",
             "How does blood pressure relate to diabetes risk?"]


class _C:
    def __init__(self, f):
        self.feature = f


def queries():
    qs = [wf.build_query([_C(a), _C(b)]) for a, b in DRIVER_PAIRS]
    qs += [wf.build_query([_C("Glucose"), _C("BMI")], q) for q in QUESTIONS[:1]]
    qs += QUESTIONS[1:]
    return qs


def retrievers():
    from ihds.vector_store import ChromaIndex

    kb = wf.KnowledgeBase(method="keyword")
    out = {"keyword": lambda q: kb.keyword_search(q, 3)}
    out["chroma-onnx"] = ChromaIndex(kb.df, embeddings="onnx").search
    if os.environ.get("OPENAI_API_KEY"):
        try:
            idx = ChromaIndex(kb.df, embeddings="openai")
            idx.search("glucose")
            out["chroma-openai"] = idx.search
        except Exception as exc:
            print(f"chroma-openai unavailable: {type(exc).__name__}: {exc}"[:200])
    return out


def compare():
    rs = retrievers()
    rows = []
    for q in queries():
        row = {"query": q}
        for name, fn in rs.items():
            row[name] = [(e["id"], e["score"]) for e in fn(q)]
        rows.append(row)
        print(f"\nQ: {q}")
        for name in rs:
            print(f"  {name:14} {row[name]}")
    names = list(rs)
    overlap = {f"{a}~{b}": round(sum(len({i for i, _ in r[a]} & {i for i, _ in r[b]}) for r in rows) / (3 * len(rows)), 3)
               for i, a in enumerate(names) for b in names[i + 1:]}
    top1 = {f"{a}~{b}": round(sum(bool(r[a]) and bool(r[b]) and r[a][0][0] == r[b][0][0] for r in rows) / len(rows), 3)
            for i, a in enumerate(names) for b in names[i + 1:]}
    print("\nmean overlap@3:", overlap, "\ntop-1 agreement:", top1)
    return {"queries": rows, "overlap_at_3": overlap, "top1_agreement": top1}


def llm_eval(n: int):
    import pandas as pd

    from ihds import agents
    from ihds.config import FEATURES
    from ihds.pipeline import build_system

    llm = agents.default_llm()
    if llm is None:
        sys.exit("OPENAI_API_KEY not set")
    out = {}
    for method in ("keyword", "chroma"):
        os.environ["IHDS_RETRIEVAL"] = method
        s = build_system(tools_mode="inprocess")
        s.kb.method = method  # the in-process tools share this KB
        g = s.graph_with_llm(llm)
        cases = s.test[s.test["Glucose"].notna() & s.test["BMI"].notna()].head(n)
        res_rows = []
        for _, r in cases.iterrows():
            rec = {f: (None if pd.isna(r[f]) else float(r[f])) for f in FEATURES}
            t0 = time.time()
            res = agents.run(g, rec)
            res_rows.append({"retrieval": res.retrieval, "draft_source": res.draft_source,
                             "approved_first": int(res.revisions == 0 and res.draft_source == "llm" and res.review["approved"]),
                             "approved": int(res.review["approved"]), "escalated": int(res.escalated),
                             "revisions": res.revisions, "grounding": res.review["grounding_score"],
                             "evidence": [e["id"] for e in res.evidence], "latency_s": round(time.time() - t0, 2)})
        df = pd.DataFrame(res_rows)
        out[method] = {"runs": len(df), "retrieval_used": sorted(df["retrieval"].unique().tolist()),
                       "llm_drafts_final": int((df["draft_source"] == "llm").sum()),
                       "reviewer_approved_first_try": round(float(df["approved_first"].mean()), 3),
                       "reviewer_approved_final": round(float(df["approved"].mean()), 3),
                       "escalated": int(df["escalated"].sum()), "mean_revisions": round(float(df["revisions"].mean()), 3),
                       "mean_grounding": round(float(df["grounding"].mean()), 3), "min_grounding": round(float(df["grounding"].min()), 3),
                       "mean_latency_s": round(float(df["latency_s"].mean()), 2)}
        print(method, out[method])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="live gpt-4o-mini eval on N cases per retriever")
    ap.add_argument("--repeats", type=int, default=1, help="repeat the live eval (LLM output varies run to run)")
    a = ap.parse_args()
    result = compare()
    if a.llm:
        runs = [llm_eval(a.llm) for _ in range(a.repeats)]
        result["llm_eval"] = runs[0] if len(runs) == 1 else {"repeats": runs}
    path = ROOT / "reports" / "retrieval_comparison.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    if not a.llm and "llm_eval" in old:
        result["llm_eval"] = old["llm_eval"]
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")
