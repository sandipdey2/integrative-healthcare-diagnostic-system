# Testing — final test plan and results

Run date: 2026-11-24. Offline suite runs with no API key; the live suite used a Vocareum key
(`OPENAI_API_BASE=https://openai.vocareum.com/v1`).

### Environments and pinned versions

`requirements.txt` installs **from binary wheels only** (`pip install --only-binary=:all: -r requirements.txt`)
on Python 3.11, 3.12 and 3.14 (Streamlit Community Cloud ignores the 3.11 setting and runs 3.14.10).

| Python | wheels-only install | offline pytest | live `vocareum` | headless Streamlit |
|---|---|---|---|---|
| 3.14.10 | ✅ (uv and pip) | 52 passed, 4 skipped | 4 passed (full suite 56/56 with key) | health `ok`; UI run shows "agent tools via mcp-stdio" |
| 3.12 | ✅ | 52 passed, 4 skipped | — | — |
| 3.11 (also Docker `python:3.11-slim`) | ✅ | 52 passed, 4 skipped | 4 passed (full suite 56/56 with key) | container healthy (MCP HTTP + API + Streamlit) |

Pins: streamlit 1.67.0, evidently 0.7.23, pandas 2.3.3, numpy 2.3.5, pillow 11.3.0, scikit-learn 1.7.2,
scipy 1.16.3, pyarrow 26.0.1, openai 3.58.0, langgraph 1.2.24, langchain-core 1.8.4, langchain-openai 1.8.5,
mcp 1.30.0, langchain-mcp-adapters 0.3.1, fastapi 0.165.0, uvicorn 0.54.0, httpx 0.28.1, prometheus-client 0.24.1
(added 2026-11-25; the 2026-11-25 additions were tested on Python 3.11 only, see §7).
`mcp` stays < 2: mcp 2.x renamed FastMCP and langchain-mcp-adapters 0.3.1 fails to import against it.
pandas stays < 3 because evidently 0.7.23 imports a pandas-2 internal. Re-running `python -m ihds.pipeline` on
3.14 with these pins reproduces every reported number (only Welch p-values differ in the 15th significant digit).

```bash
pip install -r requirements-dev.txt
pytest -q              # -> 52 passed, 4 skipped (no key)
pytest -m vocareum -v  # -> 4 passed (live, key exported)
python -m ihds.pipeline
```

## 1. Automated tests (pytest) — 56 / 56 passed with the key (52 offline + 4 live)

The `system` fixture builds the graph with its tools served by the real MCP server (stdio subprocess), so every
graph-level test below also exercises MCP.

| File | Tests | Covers | Result |
|---|---|---|---|
| `tests/test_data_stats.py` | 4 | dataset shape (768×9, 268 positive), impossible-zero cleaning, stratified split (614/154), Welch t-test ranking | 4 passed |
| `tests/test_model.py` | 4 | test ROC-AUC > 0.78 and recall > 0.6, monotonic in glucose, contributions sum exactly to the logit, imputation of missing optional values | 4 passed |
| `tests/test_gan.py` | 3 | hand-written backprop vs finite differences, valid/finite/non-collapsed samples, glucose mean within 10 mg/dL after training | 3 passed |
| `tests/test_workflow.py` | 10 | guard (ranges, required fields, injection, dosing, documented paraphrase gap), zero normalisation, retrieval (Chroma, or TF-IDF when unavailable), grounding score, grounding against evidence + case data (a real 0.185 draft passes, invented-claim drafts fail), template passes review, reviewer rejects unsafe text | 10 passed |
| `tests/test_agents.py` | 20 | **graph structure** equals the exported `docs/architecture.mmd`; **routing** functions and supervisor order; non-borderline, borderline (uncertainty agent) and blocked paths; **revision loop cap** (3 writer attempts → escalation); loop recovery after one rejection; LLM error fallback; gpt-4o-mini selection and `OPENAI_API_BASE` (Vocareum) honored; **each agent node** (safety_guard, risk, evidence, uncertainty, writer, reviewer, finalize) | 20 passed |
| `tests/test_monitoring.py` | 1 | Evidently holdout/shifted/synthetic reports written (HTML+JSON); no drift on holdout; Glucose drift and lower precision on the shifted set | 1 passed |
| `tests/test_mcp_api.py` | 9 | **real MCP server** subprocess over stdio: `list_tools` = 5 tools, `call_tool` results match the model; **streamable HTTP** server: every tool over MCP returns the same JSON as in-process; **graph via MCP** = in-process on 10 records, transport in trace and tool log; unreachable MCP → labelled `inprocess-fallback` in auto mode, error in `IHDS_TOOLS=mcp`; **FastAPI** `/health`, `/assess` (ok, unsafe question blocked, missing Glucose blocked), 422 on bad types, 400 for `use_llm` without key, `/monitoring/summary`; **Streamlit with `IHDS_API_URL`** calls `POST /assess` | 9 passed |
| `tests/test_e2e.py` | 2 | full pipeline end-to-end incl. Evidently agent evaluation; Streamlit app runs and the "Run advisory workflow" button works with no API key (`streamlit.testing.v1.AppTest`) | 2 passed |

| `tests/test_vocareum_live.py` (marker `vocareum`) | 4 | **live** "vocareum test": base URL set, gpt-4o-mini ping (`pong`, model `gpt-4o-mini-2024-07-18`), full graph run uses the LLM writer (not template) and the reviewer approves with disclaimer, unsafe request refused before any LLM call | 4 passed (live) |

Live-test note: in one full-suite run the LLM draft for the original fixed record failed grounding three times and was
escalated (the safety design working, but a flaky test). The test now uses held-out record #0 and allows up to 3
independent graph runs; after the change: `pytest -m vocareum` 4/4 passed three times in a row. After the MCP change
the live tests run with tools over MCP stdio; full suite with key 56/56 on 3.11 and 3.14.

Offline, the LLM writer path is also covered with LangChain's `FakeListChatModel` and a broken client.

## 2. Scenario evaluation (`python -m ihds.pipeline` → `evaluation/results.json`)

| Scenario | Expected | Observed | Pass? |
|---|---|---|---|
| 154 held-out records through the graph | runs complete; incomplete records blocked | 151 released, 3 blocked (missing BMI) | ✅ |
| Reviewer on released drafts | all approved | 151/151 approved, mean grounding 0.551 | ✅ |
| Risk bands are ordered | observed diabetes rate rises with band | low 9.5% (6/63), moderate 47.2% (17/36), high 59.6% (31/52) | ✅ |
| Borderline routing | every 0.40–0.60 case visits `uncertainty_agent` | 27 borderline, 27 uncertainty runs | ✅ |
| 500 GAN synthetic cases through the graph | no crashes; reviewer approves | 500/500 run, 0 blocked, 500 approved, 90 borderline | ✅ |
| Unsafe questions (5) | all blocked | 4/5 blocked | ❌ known gap |
| Benign questions (3) | all allowed | 3/3 allowed | ✅ |
| Revision-loop cap with an always-unsafe stand-in LLM | ≤3 writer attempts, then template + escalation | 3 attempts, 3 rejections, template released, `escalated=True` | ✅ |
| Docker image | builds; `/_stcore/health` = ok; container healthcheck healthy | built (`python:3.11-slim`), health `ok`, status `healthy` | ✅ |
| Docker: three layers | MCP HTTP → API → UI | `GET :8000/health` → `tools_transport: mcp-http`, 5 tools; `POST /assess` → `ok`, p 0.737, imputed SkinThickness/Insulin; UI trace "Workflow route: REST API http://127.0.0.1:8000 · tool transport: mcp-http" (`docs/screenshots/07_docker_api_mcp_trace.png`); ~0.83 GB RAM | ✅ |
| Results unchanged by MCP | `python -m ihds.pipeline` numbers identical | identical (Welch p-values differ only in the 15th digit) | ✅ |
| Headless Streamlit | serves on 8501 without a key | HTTP 200, health `ok` | ✅ |

## 2b. Live LLM evaluation (`scripts/run_llm_eval.py --n 30`, gpt-4o-mini via Vocareum)

Same 30 held-out cases (first 30 with Glucose and BMI present), run through the graph twice: live LLM writer and template writer.
Evidently `TextEvals` report: `monitoring/reports/llm_vs_template_report.html`; per-case drafts and traces: `evaluation/llm_eval_cases.csv`.

| Metric | gpt-4o-mini | template |
|---|---|---|
| Released LLM draft | 28/30 (2 escalated → template) | n/a |
| Reviewer approved on first draft | 23/30 (76.7%) | 30/30 |
| Approved after 1 revision | 5/30 | — |
| Reviewer rejections (all for grounding < 0.2) | 11 | 0 |
| Final approved | 30/30 | 30/30 |
| Drafts with `[K#]` citation / disclaimer / no forbidden phrases | 100% / 100% / 100% | 100% / 100% / 100% |
| Mean citations per draft | 2.7 | 3.0 |
| Mean grounding (min) | 0.303 (0.200) | 0.549 (0.392) |
| Mean length / sentences | 771 chars / 6.6 | 1001 chars / 8.8 |
| Mean latency per case | 4.7 s | 0.01 s |

Evidently drift test LLM vs template: grounding score and citation count distributions differ (p < 0.001).
Observation: one live draft cited [K4] (BMI guidance) for the pregnancies driver — a valid id on the wrong claim,
which the reviewer's checks cannot detect.

## 3. Evidently AI results (`monitoring/reports/`)

| Report | Key numbers |
|---|---|
| holdout (train vs test) | 0/8 features drifted; accuracy 0.734, precision 0.603, recall 0.704, F1 0.650 |
| shifted (+25 mg/dL glucose, simulated) | Glucose drifted (K-S p < 0.001); accuracy 0.682, precision 0.527, recall 0.907, F1 0.667 |
| synthetic (train vs GAN) | 5/8 drifted: Pregnancies, SkinThickness, Insulin, DiabetesPedigreeFunction, Age; Glucose, BloodPressure, BMI not drifted |
| agent_text_eval (151 drafts) | citation rate 1.0, disclaimer rate 1.0, forbidden-phrase-free rate 1.0, mean length 1014 chars; agent accuracy 0.729 / precision 0.603 / recall 0.704 |
| agent_output_drift (holdout vs shifted runs) | only `probability` drifted (p = 0.0009); mean 0.452 → 0.606; grounding, revisions, tool calls, length stable |
| guard_eval (19 labelled cases) | accuracy 0.947, precision 1.0, recall 0.889; missed: "Should she stop taking my medication?" |

## 4. Failure cases and unexpected behaviour

1. **Guard bypass by paraphrase.** The keyword pattern `stop (taking|my) medication` does not match
   "stop taking my medication". Template mode stays safe (the draft never gives medication advice), but the
   request should have been refused. Fix options: broader patterns or an LLM/embedding intent classifier.
2. **Silent sensor drift.** A +25 mg/dL calibration error raises recall but drops precision from 0.603 to 0.527
   (more false alarms). Evidently catches it in both the input drift report and the agent-output drift report.
3. **Confident false negative.** A diabetic test patient with Glucose 78, BMI 31, age 26 got p = 0.077.
   Low glucose dominates the linear model; this is why every output is advisory.
4. **GAN fidelity.** The generator does not reproduce the median-imputation spikes in Insulin/SkinThickness;
   synthetic rows are therefore used only for stress tests and the uncertainty check, never for training.

## 5. Manual checks before submission

- [x] Streamlit Cloud URL opens and "Run advisory workflow" returns a draft (https://ihds-diabetes-advisor.streamlit.app/; live LLM run verified 2026-11-25 — see §6). Unauthenticated curl of `/_stcore/health` returned HTTP 303 to Streamlit's app-auth gate, so that health check was not used as evidence.
- [x] Azure Container App URL opens; `curl https://ihds-app.bluebay-94b3aae3.westus.azurecontainerapps.io/_stcore/health` returns `ok` (HTTP 200, verified 2026-11-25).
- [x] Live: with the Vocareum key set, the draft source shows `llm` and the reviewer can approve or reject it (verified 2026-11-24 in the 30-case eval and 2026-11-25 on both live hosts; outcomes vary run to run).

## 6. Deployment smoke tests (2026-11-25)

| Target | Check | Observed | Pass? |
|---|---|---|---|
| Streamlit Community Cloud, https://ihds-diabetes-advisor.streamlit.app/ | In-browser "Run advisory workflow" on default inputs with the live LLM (graph in-process, tools via MCP stdio) | First gpt-4o-mini draft rejected (grounding 0.19); second approved (0.207); draft source `llm`. LLM output varies run to run. | ✅ |
| Azure Container Apps (West US), https://ihds-app.bluebay-94b3aae3.westus.azurecontainerapps.io | `GET /_stcore/health`; default-input assessment through the in-container FastAPI + MCP HTTP stack (`scripts/start_all.sh`) | Health `ok` (HTTP 200). Sidebar: "Workflow via REST API http://127.0.0.1:8000 · tools: mcp-http", "LLM key configured on the API server". Risk p = 0.604, high band, top driver Glucose; evidence K7, K2, K1. Tools `risk_score`, `explain_contributions`, `rag_search`, `population_context` over `mcp-http`. Three gpt-4o-mini drafts rejected for grounding (0.185, 0.182, 0.185); revision cap reached; template draft released (UI grounding 0.703) and case escalated to human review. Screenshots: `docs/screenshots/08_azure_container_app.png`, `docs/screenshots/09_azure_agent_trace.png`. | ✅ |

The Azure escalation is the designed fail-safe, not a host failure: the API, MCP tools and live writer all ran; the reviewer enforced the 0.2 grounding threshold and `finalize` released the template. The result-page "approved" badge is the template's review score. This matches the offline loop-cap test and the 2/30 escalations in the Vocareum evaluation (§2b).

## 7. MLOps layer and `/metrics` (2026-11-25, Python 3.11)

Full commands and results are in [`docs/MLOPS.md`](docs/MLOPS.md).

| Environment | Command | Result |
|---|---|---|
| Runtime (`requirements.txt` + dev deps, now with chromadb/onnxruntime/langsmith/sqlalchemy/psycopg) | `pytest -q -rs` | After the grounding/tracing change: **80 passed, 15 skipped** with no LangSmith or OpenAI key; **86 passed, 9 skipped** with both keys (skips: MLOps tools not installed, promtool, live Grafana). The audit log adds **7 SQLite tests** in `tests/test_audit_log.py` plus 1 live Postgres test skipped unless `DATABASE_URL` is set (and skipped if this host cannot reach Postgres). Measured on this box after the rebase: **88 passed, 15 skipped** without running the live Vocareum, live Postgres, or live LangSmith+LLM tests (10 other skips + 4 vocareum + 1 postgres). After the hot-reload, LangSmith-fallback and audit-count changes, with no LangSmith, OpenAI or Postgres settings at all: **89 passed, 16 skipped** (the one extra skip is the live LangSmith tracing test). The original 56 tests all still pass. Run with the new test files ignored (`test_metrics_agent.py`, `test_metrics_push.py`, `test_retrieval.py`, `test_tracing.py`, `test_embedded_api.py`, `test_audit_log.py`), the subset gives 57 passed, 11 skipped: the 56 plus the new grounding test in `test_workflow.py`. `test_e2e.py`'s AppTest goes through the embedded API. Since the grounding/tracing change: 1 grounding test in `test_workflow.py`, 4 tracing tests in `test_tracing.py` (1 of them live) and 1 LangSmith-link test in `test_embedded_api.py`. |
| MLOps (`requirements-mlops.txt`, separate venv) | `pytest -q tests/test_mlops.py tests/test_metrics_agent.py tests/test_metrics_push.py` | **17 passed, 2 skipped** (promtool not on PATH; live Grafana test without creds) |
| MLOps | `pip install -r requirements-mlops.txt` in a fresh venv, then `pip check` | installs with pip; "No broken requirements found" |
| Both | `ruff check .` (`ruff.toml`: E, F, W; E501/E702 ignored) | All checks passed. Fixing it removed 9 unused imports and invalid escapes from existing files; there were no behaviour changes |

What `tests/test_mlops.py` checks:

- **Configs:**
  - `dvc.yaml` stages, deps and params all exist;
  - Prometheus, compose and Grafana configs parse, every metric named in the dashboard is exposed by the API,
    and all ports bind to 127.0.0.1;
  - all compose services share the `ihds-monitoring` network, Prometheus waits for a healthy API, and the
    API binds 0.0.0.0;
  - the datasource URL comes from `${PROMETHEUS_URL}`, and every panel and target uses the `${datasource}`
    variable (no hard-coded uid);
  - `docs/ci.yml.example` has the lint, test, data-validation and docker jobs.
- **GX:** 30/30 on the shipped data; corrupted data fails the Glucose range and Age non-null expectations.
- **MLflow:** the run's params and metrics are logged; the reloaded model gives the same predictions.
- **Deepchecks:** the real split passes the gate; a leaked test set fails on Train Test Samples Mix.
- **SHAP:** Glucose ranks first, and base + ΣSHAP = log-odds.
- **Fairlearn:** the group counts sum to n, and DP diff = max − min selection rate.
- **Feast:** apply, materialize and online fetch in a temp repo; raw 0 Insulin comes back as missing.
- **Prefect:** the flow completes in task order, and a failed GX result stops it at `validate` (`Failed`).

What `tests/test_retrieval.py` checks:

- `keyword` mode is the original TF-IDF matcher (K1/K2/K4 for the glucose + BMI query);
- with `chromadb` unimportable, or a failing Chroma query, retrieval falls back to keyword and records the
  reason;
- Chroma (ONNX MiniLM) returns K1 and K2 for the glucose query and K6 first for family history, and returns 3
  cards for a question the keyword matcher misses entirely;
- a template draft built from Chroma evidence still passes the reviewer, including grounding;
- `rag_search` returns `method`, and the `evidence_agent` trace line contains `[retrieval: <method>`.

What `tests/test_tracing.py` checks:

- tracing is off without a key;
- a `[langsmith]` secrets table turns it on with project `ihds` and both tracing variables;
- an explicit `LANGSMITH_TRACING=false` is respected;
- the MCP stdio child env is an allow-list without the LangSmith or OpenAI keys;
- `run_url` returns None when tracing is off or the project lookup fails, and has the format
  `<host>/o/<tenant>/projects/p/<project>/r/<run>?poll=true`;
- offline, with a fake chat model and a callback recorder: the writer's call is an `llm` run named `writer_llm`
  (tag `attempt:1`) under the `writer_agent` node of the graph run, and its usage is summed into
  `AdvisoryResult.llm_tokens`;
- live (key only): the run is read back through the langsmith client with the band, status and draft tags and
  the metadata;
- live (LangSmith and OpenAI keys): every LLM run in the trace is `writer_llm` with tokens, the root
  `ihds-assessment` `total_tokens` covers them, and `langsmith_url` points at the run.

`tests/test_embedded_api.py` also checks that a run id and URL produced in the embedded API's thread reach the
Streamlit trace panel as **View trace in LangSmith** (AppTest).

What `tests/test_metrics_push.py` checks (stdlib only):

- the hand-written protobuf and snappy round-trip through reference decoders (and `cramjam` when installed);
- the push is off without `GRAFANA_RW_URL` or with a user but no token, and `record_assessment` is then a no-op;
- a `[grafana]` secrets table is read, and credentials are refused over plain `http://`;
- against a local HTTP receiver, it checks the remote-write headers and basic auth, the exact counter, histogram
  and heartbeat values with `service="ihds-streamlit"`, and that `repr()` hides the token;
- HTTP 500 and connection-refused errors return `False` and never raise;
- live (only with `GRAFANA_RW_*` in the env): Grafana Cloud accepts a push.

What `tests/test_audit_log.py` checks:

- with no `DATABASE_URL`, the log is a local SQLite file; `configure()` creates the table; a write stores the
  eight model features, risk band, reviewer status, draft source and route, and does not store the clinician
  question or the draft text;
- blocked runs are stored with `reviewer_status=n/a`, and rejected runs with the LangSmith run id when present;
  `recent()` returns newest first;
- a write failure is counted and does not raise;
- credentials are redacted from log strings, and a Neon host is labelled `postgres (neon)`;
- a Streamlit-style `[database] url` secret is read without being copied into `os.environ`;
- `POST /assess` writes a row, `GET /assessments/recent` returns it, and `/metrics` includes
  `ihds_audit_writes_total`;
- live (only with `DATABASE_URL`): a write/read against that Postgres. If this host cannot reach it, the
  test is skipped (on this box, Neon :5432 TLS timed out; the project itself responded through the Neon API).

What `tests/test_embedded_api.py` checks (FastAPI inside the Streamlit process):

- the gate: on by default, off with `IHDS_EMBED_API=0` or when `IHDS_API_URL` is set;
- the embedded uvicorn answers `/health` (on the shared `System`, `inprocess` tools in the test) and `POST /assess`;
- the pusher, against a local receiver, sends `ihds_http_requests_total{route="/assess",method="POST"}` with
  `service="ihds-streamlit"` and a 0 sample first. It sends no `_created` series, one `ihds_app_up` and no
  duplicate label sets;
- the Streamlit app (AppTest) shows `Workflow via REST API http://127.0.0.1:...` and an assessment works, with
  `(embedded)` and the retrieval in the trace caption.

What `tests/test_metrics_agent.py` checks (stdlib only, so it runs in the runtime venv):

- with no `GRAFANA_RW_URL` the agent refuses to start, and `start_all.sh` guards on it;
- the Grafana Cloud config uses `basic_auth.password_file`, never contains the token, keeps only `ihds_.*|up`,
  and drops `ihds_.*_created`;
- the token file is mode 600;
- with no user there is no `basic_auth`, and an empty token with a user fails;
- the rendered config passes `promtool check config --agent` (skipped without promtool).

End-to-end runs on the box:

| Tool | Run | Observed |
|---|---|---|
| DVC | `dvc repro -f`, `dvc status`, `dvc push` (local folder remote) | 7/7 stages ran; "Data and pipelines are up to date"; cached outputs pushed |
| Prefect | `python -m ihds.mlops.flow` | flow `Completed`, 7/7 tasks `Completed`, 37 s |
| MLflow | DVC train stage | run `8b95d9818eae4b4d94177879cdde1f3c`: ROC-AUC 0.8126, CV 0.8444 ± 0.0161 |
| GX | validate stage | 30/30 passed |
| Deepchecks | evaluate stage | gate passed, 0 critical failures (non-critical: Feature Label Correlation Change, Simple Model Comparison) |
| SHAP / Fairlearn | explain stage | top 3: Glucose, BMI, Pregnancies; DP diff 0.420, EO diff 0.563 across age bands |
| Feast | features stage | patient 0 and 5 fetched online, matching the source; p = 0.826 / 0.2059 |
| Docker | `docker compose -f docker-compose.monitoring.yml build`, then `docker run --network host ihds:local` (default CMD = `scripts/start_all.sh`) | image built; Streamlit `/_stcore/health` `ok`; API `/health` showed `tools_transport: mcp-http`; `/metrics` served 18 `ihds_` lines; container healthcheck `healthy`; `mlflow` is not in the image |
| Prometheus + Grafana | unmodified `docker-compose.monitoring.yml` on a bridge network (after `scripts/fix_docker_bridge_forward.sh`, box only), 104 `/assess` calls | `ihds-api up http://api:8000/metrics`; ok 95, blocked 9; high 35, low 33, moderate 27; escalations 0; p95 0.0486 s; Grafana datasource health OK; every panel returned series; `docs/screenshots/10_grafana_dashboard.png` |
| CI `monitoring-stack` job (by hand) | the job's steps from `docs/ci.yml.example` | `compose config -q` and `promtool check config` OK; `up --wait` with 3/3 healthy; targets `{'ihds-api': 'up', 'prometheus': 'up'}`; `sum(ihds_predictions_total)` = 20; Grafana health, datasource health and dashboard search OK |
| Metrics push (Grafana Cloud path) | `deploy/monitoring/docker-compose.push-test.yml` (same image, local receiver in place of Grafana Cloud), 31 `/assess` calls | agent log line present; token absent from config and from the agent, uvicorn and Streamlit environments; token file 600; receiver had `ihds_predictions_total{band="high",service="ihds",deployment="local-push-test"}` 30 and `ihds_assess_total` ok 30 / blocked 1; only `ihds_*` + `up` names; all 8 panels (dashboard version at the time) had data in the option B Grafana image |
| Streamlit → Grafana Cloud | `app.py` through `streamlit.testing.v1.AppTest` with the real credentials in `st.secrets`, 12 assessments; plus `streamlit run` with a user-level `secrets.toml` outside the repo, 3 clicks in headless Chrome | 11 ok, 1 blocked; high 5, low 4, moderate 2; 5 pushes HTTP 200, 0 failed; the server log showed the push line and no failures; no `GRAFANA_RW_*` in the Streamlit or MCP process environments. Wrong token gave 401; malformed body, bad metric name and future timestamp gave 400, so 200 means validated. **Read-back not possible:** the token lacks read scope (`/api/v1/query` gave 401 `invalid scope requested`) |
| Dashboard with both sources | `ihds_api.json` imported via `/api/dashboards/import` into Grafana 12.1, fed by `metrics_push` → local Prometheus receiver (40 synthetic assessments) | import OK; `service` variable = `ihds-streamlit`; 9/10 panels with data (the API-only request-rate panel was empty as expected); p50 0.03 s, p95 0.085 s |
| ChromaDB retrieval vs keyword | `scripts/compare_retrieval.py --llm 30 --repeats 2` (and one earlier `--llm 30` run) | 11 queries: overlap@3 keyword~ONNX 0.788, keyword~OpenAI embeddings 0.758, ONNX~OpenAI 0.818; top-1 agreement 0.636 / 0.545 / 0.727. Live gpt-4o-mini, 3 runs × 30 cases: first-try approval keyword 0.733 / 0.733 / 0.667 vs chroma 0.600 / 0.733 / 0.733; final approval 1.0 for both; escalated 13 vs 11 of 90; mean grounding 0.33–0.35 vs 0.29–0.31 (min 0.2). Results in `reports/retrieval_comparison.json` |
| Chroma on Python 3.14 | fresh `uv` venv, `pip install -r requirements.txt`; tests for retrieval, tracing, push, MCP and the Streamlit app | `uv pip check` clean (158 packages); 24 passed, 2 skipped; chroma answered with ONNX MiniLM |
| Docker image | `docker build` (ONNX model baked in as `appuser`), `/assess` on the container | image 2.9 GB; `retrieval: chroma (onnx-all-MiniLM-L6-v2)` over `mcp-http`; no model download at run time; Streamlit health `ok` |
| LangSmith | one MCP-stdio assessment with gpt-4o-mini and tracing on, then `Client.list_runs(project_name="ihds", trace_id=...)` | root `ihds-assessment` with tags `band:high, status:ok, draft:llm, retrieval:chroma` and matching metadata; 21 runs in the trace (16 chain, 1 llm, 4 tool); no orphan root runs; run link resolved |
| Grafana stat panels (v3) | dashboard imported into Grafana 12.1; 5 assessments pushed by `metrics_push` across a simulated process restart | stat panels 5 / 0 / 3 / 1 (assessments / escalations / high / blocked), exact; the old `sum(ihds_assess_total)` showed 3 |
| Agent off | `docker run ihds:local` with no `GRAFANA_RW_URL` | only MCP, uvicorn and Streamlit running; no `/tmp/ihds-agent` |
| Config lint | `promtool check config` (3.5.0) on `monitoring/prometheus/prometheus.yml`, `deploy/monitoring/prometheus/prometheus.yml` and the rendered agent config (`--agent`); `docker compose config -q` on both compose files; Grafana 12.1 started with the provisioning | all SUCCESS; Grafana logged no provisioning errors and listed the `ihds-api` dashboard; `ihds_api.json` imported through `/api/dashboards/import` into a Grafana whose datasource has another uid |

**Docker networking on the box:** user-defined bridge networks failed on the box because of a stale
iptables-legacy `FORWARD` chain (policy `DROP`, rules only for `docker0`). The kernel evaluates it alongside
the nft rules that dockerd writes. `scripts/fix_docker_bridge_forward.sh` adds `br-+` ACCEPT rules to the legacy
`DOCKER-USER` chain. It is a box-only fix and does not survive a reboot. Before the fix, the unmodified file
showed `ihds-api down ... context deadline exceeded`. After it, with no compose change, the target was up.
See docs/MLOPS.md.

**Not verified:** a push to a real Grafana Cloud stack and any Azure deployment of the push or option B.

**CI:** `.github/workflows/ci.yml` (added through the web UI in `890ddc9`) ran on GitHub as Actions run
45556540515 (2026-11-25 05:52 UTC). lint, test, data-validation and docker passed, and deploy was skipped. The
`monitoring-stack` job exists only in `docs/ci.yml.example` so far. Its steps were run by hand on the box (see the
table above).
