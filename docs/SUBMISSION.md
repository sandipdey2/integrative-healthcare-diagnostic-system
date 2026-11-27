# Submission guide: IHDS (diabetes-risk advisory system, advisory only)

Repository: https://github.com/sandipdey2/integrative-healthcare-diagnostic-system (branch `main`).
Status as of 2026-11-25. Every claim below links to a file in this repo. Results that depend on an LLM vary from run to run.

## 1. Live links

| What | URL | Status (2026-11-25) |
|---|---|---|
| Streamlit Community Cloud app | https://ihds-diabetes-advisor.streamlit.app/ | **Live.** It serves `main`, embeds the FastAPI app on 127.0.0.1 and runs the tools over MCP stdio. The live check at about 03:08 PST approved the first gpt-4o-mini draft (grounding 0.814) with a direct LangSmith link, and wrote a Neon audit row. |
| Azure Container Apps (West US) | `ihds-app.bluebay-94b3aae3.westus.azurecontainerapps.io` | **Offline.** It was deployed on 2026-11-25 (revision `ihds-app--0000002`, which also served `/metrics` inside the container). The Vocareum lab has since reset, and public DNS returned NXDOMAIN at 03:28 PST. The evidence from when it was live is in `TESTING.md` §6 and `docs/screenshots/08_azure_container_app.png` and `09_azure_agent_trace.png`. To bring it back, start a new lab session and run `deploy/azure/deploy_aca.sh`. |
| CI | GitHub Actions, `.github/workflows/ci.yml` | The latest run on `main` succeeded: lint, test, data-validation and docker passed; deploy was skipped. |
| Observability (optional) | LangSmith project `ihds`; Grafana Cloud (remote write); Neon Postgres project `ihds` | Configured on Streamlit Cloud through secrets. The Grafana token is write-only, so the pushed series were confirmed by HTTP 200 responses, not by reading them back. |

## 2. Run it

**Locally** (Python 3.11 to 3.14):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                    # 89 passed, 21 skipped with no keys (MLOps tests skip here; run them in the MLOps venv)
streamlit run app.py --server.headless true  # http://localhost:8501 (embedded API + MCP stdio tools)
```

**Docker** (MCP HTTP on :8765, then FastAPI on :8000, then Streamlit on :8501):

```bash
docker build -t ihds . && docker run -p 8501:8501 -p 8000:8000 ihds
curl -s localhost:8000/health
```

**Monitoring stack** (Prometheus, Grafana and Postgres on a bridge network):

```bash
docker compose -f docker-compose.monitoring.yml up -d --build   # Grafana on :3000, Prometheus on :9090
```

**Optional MLOps layer** (separate venv): run `pip install -r requirements-mlops.txt`, then either `dvc repro` (8 stages, including `cm_analysis`) or `python -m ihds.mlops.flow` (Prefect, one-off, still 7 tasks). For the scheduled local Prefect deployment, run `prefect server start` and then `python -m ihds.mlops.deploy` in another terminal (local only: no Prefect Cloud, work pool, Docker or Azure). See `docs/MLOPS.md`.

## 3. Secrets (all optional; placeholders only, never commit real values)

Put these in `.streamlit/secrets.toml` (git-ignored) or in Streamlit Cloud under App settings, then Secrets. For Docker, Azure or the shell, set them as environment variables.

```toml
OPENAI_API_KEY  = "<voc-... or sk-... key>"         # without it, the template writer is used
OPENAI_API_BASE = "https://openai.vocareum.com/v1"  # omit for api.openai.com

[langsmith]
LANGSMITH_API_KEY = "<lsv2_... key>"
LANGSMITH_PROJECT = "ihds"

[grafana]
GRAFANA_RW_URL   = "https://<your-stack>.grafana.net/api/prom/push"
GRAFANA_RW_USER  = "<Prometheus instance ID>"
GRAFANA_RW_TOKEN = "<glc_... metrics:write token>"

[database]
url = "postgresql://<user>:<password>@<host>/<db>?sslmode=require"   # or the DATABASE_URL env var; without it, data/audit.db (SQLite)
```

## 4. Rubric-to-evidence map

| Criterion | Where it is implemented | Evidence |
|---|---|---|
| Problem framing and responsible scope | `README.md`, `docs/module_summary.md` | Advisory only, the clinician decides, with the disclaimer enforced by the reviewer |
| Data and statistics (Project 2) | `ihds/data.py`, `ihds/stats_analysis.py`, `data/README.md` | Welch t-tests; `tests/test_data_stats.py` |
| Interpretable model (Project 3) | `ihds/model.py` | ROC-AUC 0.813 and recall 0.704 on 154 held-out patients; `tests/test_model.py`; `evaluation/results.json` |
| Generative model (Project 5) | `ihds/gan.py` | Gradient check in `tests/test_gan.py`; fidelity limits from Evidently |
| Multi-agent orchestration (Project 6) | `ihds/agents.py` (LangGraph) | `docs/architecture.png` is exported from code; `tests/test_agents.py` checks that the graph matches it |
| RAG and grounding | `ihds/vector_store.py` (ChromaDB, with TF-IDF as the fallback), `ihds/workflow.py` (the reviewer scores against the cards plus case data at ≥ 0.4) | `reports/grounding_eval.json`; `reports/retrieval_comparison.json`; `docs/MLOPS.md` (reviewer grounding section) |
| Tools over a real protocol | `mcp_server/server.py` (MCP stdio and HTTP), `ihds/tool_backends.py` | `tests/test_mcp_api.py` starts the real server |
| API | `api/main.py` (`/assess`, `/health`, `/assessments/recent`, `/monitoring/summary`, `/metrics`), `ihds/embedded_api.py` | `tests/test_mcp_api.py`, `tests/test_embedded_api.py` |
| UI | `app.py` | Streamlit AppTest in `tests/test_e2e.py`; `docs/screenshots/` |
| Safety | `workflow.input_guard`, reviewer, revision cap with escalation | Guard accuracy 0.947 on 19 labelled cases (1 paraphrase miss); loop-cap test |
| Persistence and audit | `ihds/audit_log.py` (Postgres/Neon or SQLite; the eight features plus run metadata, no free text) | `tests/test_audit_log.py`; Neon read-back in `TESTING.md` §5 |
| LLM observability | `ihds/tracing.py` (LangSmith root `ihds-assessment` with a `writer_llm` child) | `tests/test_tracing.py`; live run in `TESTING.md` §6 |
| Monitoring | `ihds/monitoring.py` (Evidently), `/metrics`, `ihds/metrics_push.py`, `monitoring/` | `monitoring/reports/`; Grafana dashboard screenshot `docs/screenshots/10_grafana_dashboard.png` |
| MLOps | `ihds/mlops/`, `dvc.yaml`, `feature_repo/` (GX, MLflow, DVC, Deepchecks, SHAP/Fairlearn, Feast, Prefect) | `reports/*.json`; `docs/MLOPS.md` |
| Testing and CI | `tests/`, `.github/workflows/ci.yml`, `ruff.toml` | 89 passed / 16 skipped with no keys; 95 / 10 with the LLM and LangSmith keys; MLOps venv 17 / 2; CI green |
| Deployment | `Dockerfile`, `scripts/start_all.sh`, `deploy/azure/` | Streamlit Cloud is live. Azure was deployed and is now offline (see §1) |
| Documentation | `README.md`, `diagrams/system_architecture.md`, `docs/MLOPS.md`, `TESTING.md`, `FACTS.md` | This file |

## 5. Known limitations

- **Data:** one small dataset from one population (768 Pima women aged 21 and over). The results may not transfer to other groups, and there is no external or subgroup validation.
- **Safety guard:** keyword and regex based, so a paraphrase can get past it (1 miss in 19 labelled cases).
- **Grounding:** the reviewer measures token overlap, not entailment, so a valid `[K#]` citation on the wrong claim still passes. The 0.4 threshold was calibrated on one 15-case batch per retriever plus hand-written invented drafts. The earlier 30-case live evaluation used the old cards-only 0.2 rule.
- **Routing:** the supervisor is rule-based. The LLM only writes the draft text.
- **No authentication:** the MCP server and the API have none (use them on localhost or a private network only). There is no EHR integration.
- **Deployments:** the Azure deployment is offline since the lab reset. On Streamlit Cloud, a cold start or the first load after a code push retrains the model.
- **Data pipeline:** batch only (DVC, GX, Prefect flow with a local weekly schedule that fires only while the serve process runs, Feast file source). There is no streaming and no lakehouse, by design, for a static CSV.
- **Grafana Cloud:** the pushed series were not read back, because the token is write-only.
