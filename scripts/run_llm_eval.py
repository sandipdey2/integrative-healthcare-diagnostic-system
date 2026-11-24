"""Live evaluation: gpt-4o-mini (via OPENAI_API_BASE, e.g. Vocareum) vs template drafts on N held-out cases.

    export OPENAI_API_KEY=<voc-... key>; export OPENAI_API_BASE=https://openai.vocareum.com/v1
    python scripts/run_llm_eval.py --n 30

Writes monitoring/reports/llm_vs_template_report.{html,json}, llm_eval_summary.json and
evaluation/llm_eval_cases.csv (drafts + review results; no credentials).
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ihds import agents, monitoring  # noqa: E402
from ihds.config import FEATURES, TARGET  # noqa: E402
from ihds.pipeline import build_system  # noqa: E402


def row(res, case_id, label, latency):
    return {"case_id": case_id, "label": label, "status": res.status, "probability": res.probability,
            "band": res.band, "borderline": res.borderline, "draft_source": res.draft_source,
            "grounding_score": res.review.get("grounding_score"), "approved": int(res.review.get("approved", False)),
            "revisions": res.revisions, "escalated": int(res.escalated),
            "n_citations": len(re.findall(r"\[K\d+\]", res.draft)), "latency_s": round(latency, 2),
            "failed_checks": ",".join(k for k, v in res.review.get("checks", {}).items() if not v), "draft": res.draft,
            "trace": " | ".join(res.trace)}


def main(n: int):
    llm = agents.default_llm()
    if llm is None:
        sys.exit("OPENAI_API_KEY not set")
    s = build_system()
    g_llm, g_tpl = s.graph_with_llm(llm), s.graph
    cases = s.test[s.test["Glucose"].notna() & s.test["BMI"].notna()].head(n)
    llm_rows, tpl_rows = [], []
    for i, r in cases.iterrows():
        rec = {f: (None if pd.isna(r[f]) else float(r[f])) for f in FEATURES}
        t0 = time.time(); res_t = agents.run(g_tpl, rec); lt = time.time() - t0
        t0 = time.time(); res_l = agents.run(g_llm, rec); ll = time.time() - t0
        tpl_rows.append(row(res_t, int(i), int(r[TARGET]), lt))
        llm_rows.append(row(res_l, int(i), int(r[TARGET]), ll))
        print(f"case {i}: p={res_l.probability:.2f} llm={res_l.draft_source} approved={res_l.review['approved']} "
              f"rev={res_l.revisions} grounding={res_l.review['grounding_score']}")
    ldf, tdf = pd.DataFrame(llm_rows), pd.DataFrame(tpl_rows)
    summary = monitoring.run_llm_vs_template_eval(ldf, tdf)
    out = pd.concat([ldf.assign(writer="gpt-4o-mini"), tdf.assign(writer="template")])
    (ROOT / "evaluation").mkdir(exist_ok=True)
    out.to_csv(ROOT / "evaluation" / "llm_eval_cases.csv", index=False)
    print(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    main(ap.parse_args().n)
