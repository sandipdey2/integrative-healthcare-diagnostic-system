"""Live reviewer/grounding batch: real gpt-4o-mini drafts for N cases with each retriever (keyword, chroma).

    python scripts/eval_grounding.py --label after --n 15

Per retriever it reports first-try approval of the LLM draft, how often an LLM draft was released, escalations,
and the mean grounding of every LLM draft: `grounding` as the reviewer computes it now, and `grounding_evidence_only`,
the earlier definition (share of draft tokens found in the evidence cards alone), so runs stay comparable.
Writes reports/grounding_eval.json under the given label (no credentials).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CASE = {"Pregnancies": 2, "Glucose": 140, "BloodPressure": 72, "SkinThickness": 29, "Insulin": 125,
                "BMI": 33.0, "DiabetesPedigreeFunction": 0.45, "Age": 41}  # app.py manual-entry defaults


class RecordingLLM:
    """Pass-through wrapper that keeps every draft the writer gets back."""

    def __init__(self, llm):
        self.llm, self.drafts = llm, []

    def invoke(self, msgs, *a, **k):
        r = self.llm.invoke(msgs, *a, **k)
        self.drafts.append(r.content.strip())
        return r


def main(n: int, label: str):
    import pandas as pd

    from ihds import agents
    from ihds import workflow as wf
    from ihds.config import FEATURES
    from ihds.pipeline import build_system

    base = agents.default_llm()
    if base is None:
        sys.exit("OPENAI_API_KEY not set")
    s = build_system(tools_mode="inprocess")
    held = s.test[s.test["Glucose"].notna() & s.test["BMI"].notna()].head(n - 1)
    cases = [DEFAULT_CASE] + [{f: (None if pd.isna(r[f]) else float(r[f])) for f in FEATURES} for _, r in held.iterrows()]
    out = {}
    for method in ("keyword", "chroma"):
        s.kb.method = method  # the in-process tools share this KB
        rows = []
        for i, rec in enumerate(cases):
            llm = RecordingLLM(base)
            t0 = time.time()
            res = agents.run(s.graph_with_llm(llm), rec)
            ev_ctx = [e["fact"] for e in res.evidence]
            rows.append({
                "case": "default" if i == 0 else f"test{i}", "retrieval": res.retrieval,
                "evidence": [e["id"] for e in res.evidence], "llm_calls": len(llm.drafts),
                "approved_first": int(res.revisions == 0 and res.draft_source == "llm" and res.review["approved"]),
                "llm_draft_used": int(res.draft_source == "llm"), "escalated": int(res.escalated),
                "revisions": res.revisions, "final_grounding": res.review["grounding_score"],
                "draft_grounding": [wf.review(d, res.probability, res.evidence, case_facts=res.case_facts)["grounding_score"]
                                    if hasattr(wf, "grounding_sources") else wf.grounding_score(d, ev_ctx)
                                    for d in llm.drafts],
                "draft_grounding_evidence_only": [wf.grounding_score(d, ev_ctx) for d in llm.drafts],
                "latency_s": round(time.time() - t0, 2), "drafts": llm.drafts})
            print(method, rows[-1]["case"], rows[-1]["evidence"], "first-try", rows[-1]["approved_first"],
                  "llm used", rows[-1]["llm_draft_used"], "g", rows[-1]["draft_grounding"], flush=True)
        df = pd.DataFrame(rows)
        allg = [g for r in rows for g in r["draft_grounding"]]
        allg_ev = [g for r in rows for g in r["draft_grounding_evidence_only"]]
        out[method] = {
            "cases": len(df), "retrieval_used": sorted(df["retrieval"].unique().tolist()),
            "first_try_approval": round(float(df["approved_first"].mean()), 3),
            "llm_draft_used": round(float(df["llm_draft_used"].mean()), 3),
            "escalated": int(df["escalated"].sum()), "llm_calls": int(df["llm_calls"].sum()),
            "mean_grounding_llm_drafts": round(sum(allg) / len(allg), 3) if allg else None,
            "mean_grounding_llm_drafts_evidence_only": round(sum(allg_ev) / len(allg_ev), 3) if allg_ev else None,
            "default_case": rows[0], "rows": rows}
        print(method, {k: v for k, v in out[method].items() if k not in ("rows", "default_case")}, flush=True)
    path = ROOT / "reports" / "grounding_eval.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[label] = {"model": agents.LLM_MODEL, "n_cases": len(cases), "date": time.strftime("%Y-%m-%d"), **out}
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)} [{label}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--label", default="after")
    a = ap.parse_args()
    os.environ.setdefault("IHDS_GAN_STEPS", os.environ.get("IHDS_GAN_STEPS", "3000"))
    main(a.n, a.label)
