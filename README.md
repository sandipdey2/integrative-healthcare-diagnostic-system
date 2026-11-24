# Project 7 — Integrative Industry Synthesis
## Healthcare AI Diagnostic Support: multi-agent diabetes-risk advisory system (advisory only)

A LangGraph multi-agent system that turns eight routine measurements into an **advisory draft for a
clinician**: a calibrated risk probability, the factors driving it, cited guideline snippets, a robustness
check for borderline cases, and a full audit trace. Evidently AI monitors both the input data and the
agents' outputs. Built on the public **Pima Indians Diabetes Database** (768 records; Smith et al., 1988).

> Not a diagnostic device. Every output says so, and the clinician makes the decision.

![agent graph exported from code](docs/architecture.png)

## Prior capstone projects integrated (explicit, with file evidence)

| Prior project | Method reused | Where in this repo |
|---|---|---|
| [statistical-analysis-project](https://github.com/sandipdey2/statistical-analysis-project) (Heart Failure data) | Group descriptive stats + Welch t-test (`scipy.stats.ttest_ind(equal_var=False)`) | `ihds/stats_analysis.py`; exposed to agents as the `population_context` tool |
| [breast-cancer-classification](https://github.com/sandipdey2/breast-cancer-classification) | Stratified split, `StandardScaler`, `LogisticRegression(max_iter=1000, random_state=42)`, same metric set | `ihds/data.py`, `ihds/model.py`; `risk_score` / `explain_contributions` tools |
| [generative-mnist](https://github.com/sandipdey2/generative-mnist) | MLP Generator/Discriminator GAN, LeakyReLU(0.2), BCE, Adam(2e-4, β=(0.5, 0.999)) — re-implemented in NumPy for tabular data | `ihds/gan.py`; `synthetic_neighbors` tool + Evidently fidelity report |
| [agentic-rag-system](https://github.com/sandipdey2/agentic-rag-system) | LangGraph `StateGraph` with typed state, input guard, TF-IDF RAG over a CSV knowledge base, reviewer with grounding score, gpt-4o-mini, trace logging | `ihds/agents.py`, `ihds/workflow.py`, `data/clinical_knowledge.csv` |

New in this project: the Pima dataset, the supervisor/specialist graph design, the uncertainty agent,
the revision loop with a cap and escalation, and Evidently monitoring of data **and** agent outputs.

AI domains combined: data analysis (statistics), machine learning (logistic regression), deep generative
modelling (GAN), generative AI (optional gpt-4o-mini writer), agentic systems (LangGraph).

## How it works

1. `safety_guard` checks ranges/required fields and blocks prompt-injection or dosing/diagnosis questions → `refuse`.
2. `supervisor` (rule-based router) sends the case to `risk_agent`, then `evidence_agent`, then — only for
   borderline probabilities (0.40–0.60) — `uncertainty_agent`; each returns to the supervisor.
3. `writer_agent` drafts the note (gpt-4o-mini when `OPENAI_API_KEY` is set — via `OPENAI_API_BASE`, e.g. Vocareum — otherwise a deterministic template).
4. `reviewer` checks disclaimer, citations, stated probability, banned phrases and grounding. A rejected draft
   goes back to the writer with feedback, at most 2 times; then `finalize` uses the template and marks the case escalated.

Details, state schema, assumptions, tradeoffs and boundaries: [`diagrams/system_architecture.md`](diagrams/system_architecture.md).

## Key results (reproducible: `python -m ihds.pipeline`)

| Item | Result |
|---|---|
| Test split (n=154) | accuracy 0.734, precision 0.603, recall 0.704, F1 0.650, ROC-AUC 0.813 |
| Risk bands vs observed diabetes rate (test) | low 9.5% (n=63), moderate 47.2% (n=36), high 59.6% (n=52) |
| Agent drafts (151 released) | 100% cite retrieved evidence, 100% carry the disclaimer, 0 forbidden phrases, mean grounding 0.551 |
| Safety guard (19 labelled cases) | accuracy 0.947, precision 1.0, recall 0.889 — **missed** "Should she stop taking my medication?" |
| Revision loop cap | unsafe stand-in LLM → 3 writer attempts, 3 rejections, template released, `escalated=True` |
| **Live gpt-4o-mini via Vocareum (30 test cases)** | approved first try 23/30, after 1 revision 5/30, escalated 2/30 (all 11 rejections = grounding < 0.2); 100% citations + disclaimer, 0 forbidden phrases; mean grounding 0.303 (template 0.549); 4.7 s/case |
| Simulated +25 mg/dL glucose calibration error | Evidently flags Glucose drift (p<0.001); precision 0.603 → 0.527; agent mean probability 0.452 → 0.606 |
| GAN fidelity (Evidently) | 5 of 8 features drift vs real data (Insulin, SkinThickness, Pregnancies, DPF, Age) → synthetic rows used only for stress tests |

Full numbers: `evaluation/results.json`, `monitoring/reports/`, and [`TESTING.md`](TESTING.md).

## Run locally

```bash
python3 -m venv .venv          # Python 3.11–3.14 && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # 43 offline tests (+4 live tests skipped without a key)
python -m ihds.pipeline                     # evaluation + Evidently reports
streamlit run app.py --server.headless true # http://localhost:8501
```

### Live LLM via Vocareum (optional)

```bash
export OPENAI_API_KEY=<your voc-... key>            # never commit it
export OPENAI_API_BASE=https://openai.vocareum.com/v1
pytest -m vocareum -v                              # "vocareum test": 4 live tests
jupyter notebook agentic_system.ipynb              # live multi-agent walkthrough with traces
python scripts/run_llm_eval.py --n 30              # Evidently: live LLM drafts vs template drafts
streamlit run app.py --server.headless true        # sidebar shows "LLM key detected · endpoint: ..."
```

For Streamlit, the same two values can go in `.streamlit/secrets.toml` (git-ignored; see
`.streamlit/secrets.toml.example`). Without a key everything runs with the template writer.

## Deploy

- **Streamlit Community Cloud:** repo `sandipdey2/integrative-healthcare-diagnostic-system`, branch `main`,
  main file `app.py`, any Python 3.11–3.14 (Cloud's default 3.14 works; all pins have 3.14 wheels). Secrets (optional, TOML):
  `OPENAI_API_KEY = "voc-..."` and `OPENAI_API_BASE = "https://openai.vocareum.com/v1"`.
- **Azure Container Apps:** `Dockerfile` (port 8501) + [`deploy/azure/README.md`](deploy/azure/README.md)
  (`az acr build` + `az containerapp create`; ACI alternative and cleanup scripts included). With
  `WITH_OPENAI=1` the key becomes the Container Apps secret `openai-key` (`OPENAI_API_KEY=secretref:openai-key`)
  and `OPENAI_API_BASE` is set as an environment variable.

## Repository map

| Path | Content |
|---|---|
| `app.py` | Streamlit UI (risk assessment, model & data, synthetic stress test, monitoring, architecture, limits) |
| `ihds/` | `agents.py` (LangGraph), `workflow.py` (guard/RAG/reviewer blocks), `model.py`, `gan.py`, `stats_analysis.py`, `data.py`, `monitoring.py`, `pipeline.py` |
| `data/` | Pima CSV, clinical knowledge base, data card |
| `notebooks/integrated_diagnostic_system.ipynb` | Executed walkthrough of the same code |
| `tests/` | 43 offline pytest tests (units, every agent node, routing, loop cap, Evidently, end-to-end, Streamlit AppTest) + 4 live `vocareum` tests |
| `agentic_system.ipynb` | Live run of the multi-agent graph (low / borderline / high risk, refusal, revision loop) with traces |
| `scripts/` | `export_graph.py` (diagram from code), `run_llm_eval.py` (live LLM vs template, Evidently) |
| `monitoring/reports/` | Evidently HTML/JSON reports |
| `evaluation/` | `results.json`, synthetic samples |
| `docs/` | `architecture.mmd/.png` (exported from code), app screenshots, `module_summary.pdf` |
| `deploy/azure/` | Container Apps / ACI scripts and guide |

## Limitations

Single public dataset from one population; small hand-written knowledge base; keyword guard can be bypassed by
paraphrase; GAN does not reproduce imputation-heavy features; live LLM drafts are less grounded than template
drafts, and the reviewer cannot detect a valid citation attached to the wrong claim. See `TESTING.md`.
