# Verified facts for the P8 deck, and module_summary.pdf

All numbers come from `python -m ihds.pipeline` (seed 42) on 2026-11-24 → `evaluation/results.json`,
`monitoring/reports/*.json`. Re-running reproduces them exactly.

## Identity
- Title: Healthcare AI Diagnostic Support — multi-agent diabetes-risk advisory system (advisory only)
- Industry: healthcare, primary-care / screening triage. Users: clinicians. Output: advisory draft, never a diagnosis.
- Repo: https://github.com/sandipdey2/integrative-healthcare-diagnostic-system · entry `app.py` · package `ihds/`

## Data
- Pima Indians Diabetes Database, OpenML #37 (Smith et al., 1988): 768 rows, 8 features, 268 positive (34.9%).
- Females 21+, Pima heritage. Glucose = 2-h OGTT plasma glucose (mg/dL).
- Impossible zeros treated as missing: Insulin 374 (48.7%), SkinThickness 227 (29.6%), BloodPressure 35 (4.6%), BMI 11 (1.4%), Glucose 5 (0.7%).
- Split: stratified 80/20 → 614 train / 154 test.

## Prior projects → components (explicit integration)
| Prior repo | Method | Module | Agent tool |
|---|---|---|---|
| Project 2 — [statistical-analysis-project](https://github.com/sandipdey2/statistical-analysis-project) | Welch t-test | ihds/stats_analysis.py | population_context |
| Project 3 — [breast-cancer-classification](https://github.com/sandipdey2/breast-cancer-classification) | StandardScaler + LogisticRegression(max_iter=1000, rs=42), metrics | ihds/model.py | risk_score, explain_contributions |
| Project 5 — [generative-mnist](https://github.com/sandipdey2/generative-mnist) | MLP GAN, LeakyReLU 0.2, BCE, Adam 2e-4 (0.5, 0.999) → NumPy tabular GAN | ihds/gan.py | synthetic_neighbors |
| Project 6 — [agentic-rag-system](https://github.com/sandipdey2/agentic-rag-system) | LangGraph StateGraph, input guard, TF-IDF RAG, reviewer + grounding, gpt-4o-mini | ihds/agents.py, ihds/workflow.py | rag_search |
Domains: statistics/data analysis, ML, deep generative model (GAN), GenAI (optional LLM writer), agentic (LangGraph).

## Statistics (Welch, diabetic vs non-diabetic)
Glucose 142.3 vs 110.6, t=14.88, p=2.8e-41, d=1.16 · BMI 35.4 vs 30.9, p=2.5e-18, d=0.69 · Age 37.1 vs 31.2, p=1.2e-11, d=0.52 ·
all 8 features p < 1e-5.

## Model (logistic regression, class_weight=balanced, threshold 0.5)
- Test (n=154): accuracy 0.734, precision 0.603, recall 0.704, F1 0.650, ROC-AUC 0.813; TN 75, FP 25, FN 16, TP 38.
- Train (n=614): accuracy 0.759, ROC-AUC 0.851 (small gap → little overfitting).
- Odds ratio per SD: Glucose 3.27, BMI 2.03, Pregnancies 1.45, DPF 1.33, Age 1.20; Insulin/BP/Skin ≈ 1.

## GAN
- 3000 steps, batch 64, latent 16, 2×64 hidden; final D loss 1.334, G loss 0.717.
- Means real vs synthetic: Glucose 121.7 vs 119.5, BMI 32.4 vs 33.0, Age 33.4 vs 33.5, Insulin 137.7 vs 131.7.
- Evidently: 5/8 features drift vs real (Pregnancies, SkinThickness, Insulin, DPF, Age) → used only for stress tests / uncertainty, never training.

## LangGraph agents (ihds/agents.py)
- Nodes: safety_guard, supervisor, risk_agent, evidence_agent, uncertainty_agent, writer_agent, reviewer (+ refuse, finalize).
- Supervisor is rule-based (hub-and-spoke); uncertainty_agent only when 0.40 ≤ p < 0.60.
- Reviewer → writer revision loop, MAX_REVISIONS = 2 (writer ≤ 3 runs) → template + escalated=True. recursion_limit 30.
- Writer: gpt-4o-mini (langchain-openai, temperature 0) only if OPENAI_API_KEY; base URL from OPENAI_BASE_URL / OPENAI_API_BASE (Vocareum https://openai.vocareum.com/v1); else deterministic template.
- Typed state (TypedDict) with append-only trace and tool_calls (each entry: agent, tool, transport, args, preview, time).
- Diagram exported from code: docs/architecture.mmd/.png; test asserts they match. Service view docs/deployment.mmd/.png built from FastAPI routes + tools listed over MCP + agent→tool pairs from a real run.

## MCP server + FastAPI (implemented 2026-11-24)
- MCP server: mcp_server/server.py, official `mcp` SDK 1.30.0 FastMCP("ihds-tools"); 5 tools: risk_score, explain_contributions, rag_search, population_context, synthetic_neighbors (JSON in/out, null = missing). Transports: stdio (default) and streamable HTTP (`--transport streamable-http --port 8765`, path /mcp). Never receives the API key.
- Agents → MCP: ihds/tool_backends.py, langchain-mcp-adapters 0.3.1 MultiServerMCPClient + load_mcp_tools, one persistent session on a background asyncio thread. Default stdio subprocess; IHDS_MCP_URL → HTTP. IHDS_TOOLS=auto (default) falls back to labelled "inprocess-fallback" only if MCP can't start; =mcp fails instead; =inprocess explicit.
- Trace shows transport: "risk_agent: [tools via mcp-stdio] p=... ". MCP and in-process give identical outputs (test on 10 records + every tool over HTTP).
- Agent→tool map (from tool-call log): risk_agent → risk_score, explain_contributions; evidence_agent → rag_search, population_context; uncertainty_agent → synthetic_neighbors.
- FastAPI api/main.py (fastapi 0.165.0, uvicorn 0.54.0): POST /assess, GET /health, GET /assessments/recent, GET /monitoring/summary, GET /schema/features, /docs. use_llm=true without key → 400; bad types → 422.
- Streamlit uses the API when IHDS_API_URL is set; else it embeds the FastAPI app in its own process (ihds/embedded_api.py, uvicorn daemon thread on 127.0.0.1, shared System, IHDS_EMBED_API gate, in-process fallback); tools via MCP stdio. Its API metrics are pushed with service=ihds-streamlit.
- Docker: scripts/start_all.sh → MCP HTTP 127.0.0.1:8765 → API :8000 (IHDS_TOOLS=mcp) → Streamlit :8501; container healthy, /health tools_transport "mcp-http", UI trace "Workflow route: REST API ... · tool transport: mcp-http" (docs/screenshots/07_docker_api_mcp_trace.png); ~0.83 GB RAM after one UI run.
- Timing: MCP stdio graph run ≈ 0.02 s per case (50 cases 0.86 s) after ~5 s server start-up. Pipeline evaluation results unchanged by the switch to MCP (float noise only).

## Scenario evaluation
- 154 test records: 151 released, 3 blocked (missing BMI); reviewer approved 151/151; mean grounding 0.551.
- Bands vs observed diabetes: low 9.5% (6/63), moderate 47.2% (17/36), high 59.6% (31/52).
- 27 borderline cases → 27 uncertainty_agent runs. 500 GAN cases: 0 blocked, 500 approved, 90 borderline.
- Guard: 19 labelled cases → accuracy 0.947, precision 1.0, recall 0.889; MISSED "Should she stop taking my medication?".
- Loop cap: always-unsafe stand-in LLM → 3 writer attempts, 3 rejections, template released, escalated.

## Evidently
- holdout: 0/8 drift; acc 0.734 / prec 0.603 / rec 0.704 / F1 0.650.
- shifted (+25 mg/dL glucose, simulated): Glucose drift p<0.001; acc 0.682 / prec 0.527 / rec 0.907 / F1 0.667.
- agent_text_eval (151 drafts): citation 100%, disclaimer 100%, forbidden phrases 0%, mean 1014 chars.
- agent_output_drift: only probability drifts (p=0.0009), mean 0.452 → 0.606.
- guard_eval: as above.

## Failure cases (for "Evaluation and Reflection")
1. Guard paraphrase bypass (keyword brittleness). 2. Glucose calibration drift → precision 0.603→0.527, caught by Evidently.
3. Confident false negative: Glucose 78, BMI 31, age 26, diabetic, p=0.077. 4. GAN fidelity on imputation-heavy features.

## Live LLM (gpt-4o-mini-2024-07-18 via Vocareum, 2026-11-24)
- Live tests (`pytest -m vocareum`): 4/4 passed — ping, full graph with LLM writer approved + disclaimer, unsafe request refused before LLM.
- 30 held-out cases (scripts/run_llm_eval.py), re-run 2026-11-25 after the grounding fix (0919e9c; MIN_GROUNDING 0.4; Chroma; tools via mcp-stdio): first-draft approval 30/30, 0 revisions, 0 escalated; mean grounding LLM 0.729 (min 0.581) vs template 0.989 (min 0.879); citations 2.07 vs 3.0; length 721 vs 1003 chars; latency 3.7 s vs 0.15 s; drift test on grounding & citations p < 0.001. Single run; output varies.
- History, 2026-11-24 run (cards-only score, 0.2 threshold): first-draft approval 23/30 (76.7%); 5 approved after 1 revision; 2 escalated after 3 rejected drafts (template released); 11 rejections, all grounding < 0.2; final approval 30/30.
- Evidently TextEvals: citation 100%, disclaimer 100%, forbidden phrases 0 (LLM and template).
- 2026-11-24 run: mean grounding LLM 0.303 (min 0.200) vs template 0.549 (min 0.392); citations 2.7 vs 3.0; length 771 vs 1001 chars; latency 4.7 s vs 0.01 s per case; drift test LLM vs template on grounding & citations p < 0.001.
- Notebook agentic_system.ipynb re-executed live with tools via mcp-stdio: low (#8, p=0.132, approved), borderline (#2, p=0.439, uncertainty agent range 0.208–0.699; this run 3 grounding rejections → escalated), high (#7, p=0.962, approved), refusal (metformin dose), #1 approved first try this run (1 revision in the earlier run), cap (#3: 3 rejections → escalated). LLM output varies run to run.
- Failure insight: one live draft cited [K4] (BMI) for the pregnancies driver — valid id, wrong claim; reviewer can't detect.

## Live deployments (2026-11-25)
- Streamlit Community Cloud: https://ihds-diabetes-advisor.streamlit.app/ — FastAPI embedded on 127.0.0.1 (workflow_route `embedded-api`), tools via mcp-stdio, retrieval ChromaDB. Live check 2026-11-25 ~03:08 PST (commit c20199d, default inputs): the first gpt-4o-mini draft was approved (grounding 0.814, 601 tokens), with a direct LangSmith run link; the LangSmith root `ihds-assessment` has one `writer_llm` child; the audit count went 1 → 2. Neon `assessments` table read back on 2026-11-25 ~03:35 PST: 7 rows, all `embedded-api`, all with a LangSmith run id. There are 6 non-blocked LLM runs (low, moderate and high bands; all approved first try, `chroma`) plus 1 blocked refusal row. The rows store the eight model features only (no question or draft text). Earlier (before the grounding fix), a run had its first draft rejected (0.19) and the second approved (0.207).
- Azure Container Apps (West US): **offline now.** It was deployed 2026-11-25 from the Vocareum Azure lab (revision ihds-app--0000002, which also served /metrics inside the container). The lab has since reset. On 2026-11-25 at 03:28 PST, public DNS returned NXDOMAIN for the hostname. The facts below are from when it was live. Azure Container Apps (West US): https://ihds-app.bluebay-94b3aae3.westus.azurecontainerapps.io — image from `deploy/azure/deploy_aca.sh` (`scripts/start_all.sh`: MCP HTTP :8765 → FastAPI :8000 → Streamlit :8501; ingress 8501). `GET /_stcore/health` → `ok` (HTTP 200). Default inputs: sidebar "Workflow via REST API http://127.0.0.1:8000 · tools: mcp-http", "LLM key configured on the API server"; p=0.604 high, top driver Glucose; evidence K7,K2,K1; tools risk_score, explain_contributions, rag_search, population_context over mcp-http. Writer: 3 gpt-4o-mini drafts, reviewer rejected all for grounding (0.185/0.182/0.185) → revision cap → template draft released (UI grounding 0.703) and case escalated to human review. The result-page "approved" badge is the template's review, not an approved LLM draft; the banner and trace record the escalation. Screenshots: docs/screenshots/08_azure_container_app.png, 09_azure_agent_trace.png.

## MLOps layer (2026-11-25, optional, local only; details in docs/MLOPS.md)
- Separate env, `requirements-mlops.txt`: prefect 3.9 needs fastapi<0.151, while the app pins 0.165.0. The runtime adds prometheus-client 0.24.1, sqlalchemy 2.1.9 and psycopg[binary] 3.3.9 (audit log).
- GX 1.25: suite `pima_clinical_input`, 30 expectations, 30/30 passed. One BloodPressure value (24) is below the 30 floor, so ranges use mostly=0.99.
- MLflow 3.18: experiment `ihds-risk-model`, SQLite store at mlruns/mlflow.db (MLflow 3 refuses the plain file store by default).
  - Test results: acc 0.7338, prec 0.6032, rec 0.7037, F1 0.6496, ROC-AUC 0.8126; TN75 FP25 FN16 TP38.
  - CV ROC-AUC 0.8444 ± 0.0161.
  - Run IDs: `dvc repro -f` → 8b95d9818eae4b4d94177879cdde1f3c; Prefect → b1085973ece0449c93940c9b9d94725c.
- DVC 3.67: 9 stages (the 8th, `cm_analysis`, and the 9th, `drift_stats`, were added 2026-11-27). Local remote `dvc-storage/` (gitignored). Raw CSV stays in git, hashed in dvc.lock. The earlier full `dvc repro -f` ran the original 7 stages; the new stages were locked with `dvc repro cm_analysis` and `dvc repro drift_stats` only. Prefect still has 7 tasks.
- Confusion-matrix analysis (`python -m ihds.cm_analysis`): test CM TN 75 FP 25 FN 16 TP 38; sensitivity 0.704 (Wilson 0.572–0.809); AUC 0.8126 (DeLong 0.746–0.880); accuracy vs NIR exact binomial two-sided p = 0.028; McNemar vs majority two-sided p = 0.13; Youden threshold from train OOF is 0.559 (test CM 78/22/21/33, recall 0.611); 21–29 recall 0.438 (Wilson 0.231–0.668); Brier 0.181, mean predicted 0.445 vs prevalence 0.351. Details in TESTING.md §8 and reports/confusion_matrix_analysis.json.
- Drift statistics (`python -m ihds.drift_stats`): holdout has no overall alert. PSI < 0.10 on every feature except Insulin 0.241 (moderate, watch): KS p = 0.103, chi-square p = 0.116, BH-adjusted 0.822 and 0.928. Label chi-square p = 1.0. Shifted (+25 mg/dL Glucose): Glucose alert only (KS D = 0.366, p = 4.0×10⁻¹⁵, PSI 0.784, chi-square p = 1.8×10⁻¹²). A moderate PSI is a watch, not an alert, unless KS or chi-square agrees. Details in TESTING.md §9 and reports/drift_stats.json.
- Deepchecks 0.19.1 (with NumPy 2 alias shim): gate passed, 0 critical failures.
  - data_integrity: 11/12 pass.
  - train_test_validation: 8/12 pass, 1 fail (Feature Label Correlation Change, Insulin PPS diff 0.3).
  - model_evaluation: 12/18 pass, 2 warn (Weak Segments), 1 fail (Simple Model Comparison, class-0 F1 gain −0.97%).
  - The rest were not applicable.
- SHAP 0.51, mean |SHAP| in log-odds: Glucose 1.0136, BMI 0.587, Pregnancies 0.3285, DPF 0.2051, Age 0.1488, Insulin 0.0297, BP 0.0106, Skin 0.0088.
- Fairlearn 0.15: no sex/race column exists, so age bands are used (test split).
  - Selection rate: 21–29 0.235 (n=85), 30–44 0.600 (n=40), 45+ 0.655 (n=29).
  - Recall: 0.438 / 0.731 / 1.000.
  - DP diff 0.4199 (ratio 0.3591); EO diff 0.5625 (ratio 0.4375). Reported, not mitigated.
- Feast 0.68: patient 0 online features match the parquet; model p = 0.826 (patient 5: 0.2059). patient_id = CSV row; one fixed timestamp.
- Prefect 3.9.5: flow `ihds-mlops-pipeline` Completed, 7/7 tasks, 37 s, temporary local API.
- Prefect deployment: `python -m ihds.mlops.deploy` serves `ihds-mlops-pipeline/weekly-local` (cron `0 6 * * 1`, America/Los_Angeles, `with_feast: true`) with `flow.serve()` against a local `prefect server start`. Tasks retry twice (10 s apart) except on `QualityGateError`. Demo 2026-11-27: runs `f4aa95d0-936f-40c3-88eb-929a4d51f717` (21.9 s) and `fe2ae45e-0868-4b2d-a121-33ff5abe9ae1` (19.5 s) both Completed, 7/7 tasks, triggered by hand; scheduled runs not waited for. No Prefect Cloud, work pool, worker, Docker or Azure deployment; the schedule fires only while the serve process runs.
- Prometheus/Grafana: `/metrics` plus the compose stack.
  - Box result (unmodified compose, bridge network): 104 /assess calls → ok 95, blocked 9; high 35, low 33, moderate 27; escalations 0; p95 0.0486 s; target `ihds-api up`.
  - Box bridge failure root cause: stale iptables-legacy FORWARD policy DROP (no br-* rules). The box-only fix is scripts/fix_docker_bridge_forward.sh (not persistent).
  - Streamlit Cloud: ihds/metrics_push.py pushes ihds_* (service=ihds-streamlit) to Grafana Cloud via remote write (hand-encoded protobuf + literal snappy, urllib, no new deps); off unless [grafana] GRAFANA_RW_* secrets set. Box test against the real stack: all pushes HTTP 200, 0 failures; token is write-only so not read back from Grafana Cloud. Dashboard has datasource + service variables (10 panels).
  - Azure: optional in-container Prometheus agent pushes ihds_* via remote_write when GRAFANA_RW_URL/USER/TOKEN (ACA secrets) are set (deploy/azure/enable_metrics_push.sh). Verified locally against a local receiver; NOT deployed and NOT tested against real Grafana Cloud. Option B (self-hosted ACA Prometheus+Grafana) is scripted, not deployed. ACI sets IHDS_METRICS=0.
- Retrieval: rag_search uses ChromaDB 1.5.9 (EphemeralClient; ONNX all-MiniLM-L6-v2 via onnxruntime 1.32.0, no key; ~80 MB model download on first use, baked into the Docker image) over K1–K14, with TF-IDF keyword fallback (IHDS_RETRIEVAL=keyword|auto|chroma). The trace says `retrieval: chroma|keyword`. Comparison (11 queries): overlap@3 keyword~ONNX 0.788, keyword~OpenAI-embeddings 0.758, ONNX~OpenAI 0.818. Live gpt-4o-mini, 3×30 cases: first-try approval keyword 0.711 vs chroma 0.689 (spread 0.60–0.733), final approval 1.0 both, escalations 13 vs 11 of 90, mean grounding 0.33–0.35 vs 0.29–0.31. +33 packages / +281 MB on py3.14.
- LangSmith: optional (ihds/tracing.py), on only with LANGSMITH_API_KEY (env or [langsmith] secret); project `ihds`; root run `ihds-assessment` with tags band/status/draft/retrieval + metadata. Verified on the box: 21 runs in one trace (16 chain, 1 llm, 4 tool), no orphan root runs. Key not passed to the MCP child.
- Reviewer grounding (2026-11-25): root cause of LLM drafts failing grounding was the reviewer scoring against the evidence cards only, while the writer must state probability/band/drivers/population context and the disclaimer (a faithful default-case draft scored 0.185; 44/54 content words came from case data or the disclaimer). Same cards (K7,K2,K1) for keyword and Chroma, so not a Chroma regression. Fix: grounding sources = evidence cards + case-data lines, disclaimer excluded, threshold raised 0.2 → 0.4 (invented-claim drafts score 0.14–0.32 and fail), actionable reviewer feedback, writer prompt separates case data from evidence cards. scripts/eval_grounding.py, 15 cases, gpt-4o-mini: first-try approval keyword 0.600→1.000, chroma 0.667→1.000; LLM draft released 0.800/0.933→1.000; escalated 3/1→0; mean grounding (reviewer) 0.218/0.207→0.757/0.756, cards-only 0.298/0.305 after (reports/grounding_eval.json). Earlier 30-case numbers above used the cards-only score with 0.2.
- LangSmith LLM runs: each writer call is an `llm` child run `writer_llm` (under writer_agent, in the ihds-assessment trace) with tokens; root metadata llm_calls/llm_total_tokens/hostname; AdvisoryResult and POST /assess carry langsmith_run_id, langsmith_url, llm_tokens; the trace panel shows "View trace in LangSmith". Box check through the embedded API: writer_llm 592 tokens, root total_tokens 592, URL returned.
- Logs (Loki): ihds/logging_config.py writes one JSON object per line (stdlib logging, allow-listed fields only) to stdout and, with IHDS_LOG_FILE, a rotating file; api/main.py adds an X-Request-ID middleware, a `request` line per call (not /metrics; route template, status, latency_ms) and an `assessment` line per /assess (band, draft_source, escalated, revisions, blocked, guard_reason_count, trace node names). Never logged: feature values, probability, question, guard-reason text, draft, LLM prompts/outputs, secrets. docker-compose.monitoring.yml adds grafana/loki:3.7.10 (filesystem, 72 h retention) and grafana/alloy:v1.22.0 (labels job, source, level, route; request_id is not a label) plus a provisioned Loki datasource and dashboard `IHDS logs (Loki)` (13 panels). Box run outside Docker (Loki 3.7.10, Alloy v1.22.0, Grafana 12.1.0 binaries): 20 log lines written by the API (16 requests), 20 in Loki; all 13 panel queries returned results; escalation panels empty (no run escalated). The compose services were NOT run under Docker (no Docker on the box). Local only; not on Azure or Streamlit Cloud; no Grafana Cloud log push.
- 2026-11-27 Loki update: app venv 122 passed / 21 skipped (20 new in tests/test_logging.py), MLOps venv tests/test_mlops.py 15 passed, `ruff check .` clean.
- Arize AX tracing: optional (ihds/arize_tracing.py; arize-otel 0.14.1, openinference-instrumentation-langchain 0.1.90), on only when both ARIZE_SPACE_ID and ARIZE_API_KEY are in the environment (IHDS_ARIZE=0 turns it off); default project `ihds-diagnostic-agents`; one `ihds-assessment` AGENT span per agents.run() with ihds.* attributes (status, risk_band, borderline, probability rounded to 2 decimals, escalated, revisions, draft_source, retrieval, grounding_score, llm_total_tokens); inputs/outputs/prompts/completions hidden by default (IHDS_ARIZE_CAPTURE_CONTENT=1 shows them); independent of LangSmith. Box check 2026-11-27: 3 live /assess calls with gpt-4o-mini, 79 spans in 4 OTLP export calls, all SUCCESS; Arize UI not opened; no evals/monitors built; not set on Azure or Streamlit Cloud.
- Assessment audit log (ihds/audit_log.py, sqlalchemy 2.1.9 + psycopg 3.3.9): one row per completed assessment (eight model features as JSONB, risk band/probability, reviewer status, draft source, revisions, escalation, blocked flag, retrieval method, workflow route, optional LangSmith run id). No clinician question or draft text. URL from DATABASE_URL or the Streamlit `[database] url` secret; otherwise local SQLite at data/audit.db (gitignored). Writes are caught, logged with credentials redacted, and counted (`ihds_audit_writes_total`); they never fail the assessment. docker-compose.monitoring.yml includes postgres:17 (local user/password ihds/ihds, host port 127.0.0.1:5432) and sets the API's DATABASE_URL to it. Streamlit sidebar: `Audit log: postgres (neon) · N rows` or `sqlite (local)`; Audit log tab shows the last 20 rows. GET /assessments/recent?limit= lists recent rows. Neon project `ihds` (Postgres 17, aws-us-west-2, database `ihds`, role `ihds_app`) is up (checked through the Neon API). This box cannot open a Postgres TLS session to Neon on port 5432, so the live psycopg write was not run from here; Streamlit Cloud will connect once the `[database]` secret is added. Until then the Cloud app falls back to ephemeral SQLite.
- Grafana stat panels: round(sum(increase(...[$__range]))) (dashboard v3); new counter series get a 0 sample first.
- GitHub Actions: docs/ci.yml.example (lint/test/data-validation/docker/monitoring-stack/optional deploy). Active as .github/workflows/ci.yml (890ddc9, added via web UI); run 45556540515 passed lint/test/data-validation/docker, deploy skipped. monitoring-stack job only in the example until the workflow file is updated via the web UI.

## Tests
- 56 pytest tests: 52 offline (pass on py3.11, 3.12, 3.14) + 4 live `vocareum` (skipped without key); 56/56 pass with key on 3.11 and 3.14. New: 9 in tests/test_mcp_api.py (real MCP subprocess stdio list/call, HTTP server vs in-process, graph via MCP = in-process, fallback, FastAPI health/assess/blocked/422/400/monitoring, Streamlit via API). Docker image builds, healthcheck healthy. Azure Container Apps deployment smoke-tested 2026-11-25 (see Live deployments).
- 2026-11-25 MLOps update: runtime env 56 passed / 11 skipped (52 original + 1 /metrics + 3 config tests; skips are 4 live and 7 MLOps). MLOps env: tests/test_mlops.py 10/10 passed. `ruff check .` clean.
- 2026-11-25 grounding/tracing update: runtime env 80 passed / 15 skipped with no LangSmith or OpenAI key; 86 passed / 9 skipped with both keys (see TESTING.md).
- 2026-11-25 audit log: 7 SQLite tests in tests/test_audit_log.py plus 1 live Postgres test skipped unless DATABASE_URL is set (and skipped if this host cannot reach Postgres). Measured after rebase: 88 passed / 15 skipped without the live Vocareum, live Postgres, or live LangSmith+LLM tests. `ruff check .` clean.
- 2026-11-25 final readiness pass (HEAD c20199d + docs): runtime env without any keys or DATABASE_URL: **89 passed / 16 skipped**; `ruff check .` clean; latest GitHub Actions run on main succeeded (lint, test, data-validation, docker; deploy skipped).

## Honest-scope statements (use these words)
- Rule-based supervisor; LLM only writes text and is optional.
- Tools served by a real MCP server in a separate process (stdio or HTTP); the agent graph runs in one process (API or Streamlit). Per-request graph state is not a long-term memory store; completed assessments are written to an audit log (Postgres when a URL is configured, otherwise local SQLite) with the eight model features only (no question or draft text). MCP server and API have no authentication (localhost/private network only).
- Offline runs use the template writer; scenario numbers above are template runs (reproducible).
- Live gpt-4o-mini verified via Vocareum on 2026-11-24 (see Live LLM section); before the 2026-11-25 grounding fix the revision loop fired naturally with the live model; after it (threshold 0.4 vs cards + case data), 15/15 drafts per retriever were approved first try, so the loop is now shown by the offline unsafe-stand-in test and by invented drafts (scored 0.14–0.32, rejected).

## References usable in the paper (APA 7)
- American Diabetes Association Professional Practice Committee. (2024). 2. Diagnosis and classification of diabetes: Standards of Care in Diabetes—2024. *Diabetes Care, 47*(Suppl. 1), S20–S42. https://doi.org/10.2337/dc24-S002
- Amann, J., Blasimme, A., Vayena, E., Frey, D., & Madai, V. I. (2020). Explainability for artificial intelligence in healthcare: A multidisciplinary perspective. *BMC Medical Informatics and Decision Making, 20*(1), 310. https://doi.org/10.1186/s12911-020-01332-6
- Diabetes Prevention Program Research Group. (2002). Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin. *New England Journal of Medicine, 346*(6), 393–403. https://doi.org/10.1056/NEJMoa012512
- Goodfellow, I., et al. (2014). Generative adversarial nets. *Advances in Neural Information Processing Systems, 27*.
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems, 28*.
- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.
- US Preventive Services Task Force. (2021). Screening for prediabetes and type 2 diabetes: US Preventive Services Task Force recommendation statement. *JAMA, 326*(8), 736–743. https://doi.org/10.1001/jama.2021.12531
- US Food and Drug Administration. (2022). *Clinical decision support software: Guidance for industry and FDA staff*.
- Evidently AI. (2024). *Evidently documentation*. https://docs.evidentlyai.com/ · LangChain. (2024). *LangGraph documentation*. https://langchain-ai.github.io/langgraph/
