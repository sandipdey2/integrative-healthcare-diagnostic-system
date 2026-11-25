# MLOps layer — what is implemented, how to run it, and the measured results

Everything here is **optional and local-only**. The Streamlit Cloud app and the Azure Container App install
only `requirements.txt` and never import `ihds/mlops/`. The one runtime addition is `prometheus-client`, which
serves `GET /metrics` on the FastAPI service.

All numbers below come from runs on 2026-11-25 (seed 42, Python 3.11, versions pinned in
`requirements-mlops.txt`). Re-running reproduces the metrics exactly. MLflow run IDs and timings change on
every run.

| Tool | Role here | Code / config | Status |
|---|---|---|---|
| MLflow 3.18 | Experiment tracking for the risk model | `ihds/mlops/train_tracked.py` | Implemented, run locally |
| DVC 3.67 | Data + pipeline versioning (7 stages) | `dvc.yaml`, `params.yaml`, `dvc.lock`, `.dvc/config` | Implemented, `dvc repro` run locally; local remote only |
| Great Expectations 1.25 | Validation of the clinical input data | `ihds/mlops/validate.py` | Implemented, gate in DVC, Prefect and CI |
| Feast 0.68 | Local feature store (file offline store, SQLite online store) | `feature_repo/`, `ihds/mlops/features.py` | Implemented, local only; the app does not read from Feast |
| Prefect 3.9 | Orchestration of the stages | `ihds/mlops/flow.py` | Implemented, runs locally on a temporary Prefect API (no Prefect Cloud) |
| GitHub Actions | CI/CD | `docs/ci.yml.example`, active copy in `.github/workflows/ci.yml` | Active; first run passed (lint, test, data-validation, docker). The new `monitoring-stack` job is in the example only (see below) |
| FastAPI | Model-serving API | `api/main.py` | Already existed; unchanged apart from `/metrics` |
| Docker | Container image | `Dockerfile`, `scripts/start_all.sh` | Already existed; unchanged; rebuilt and smoke-tested |
| Deepchecks 0.19.1 | Model-quality testing with a pass/fail gate | `ihds/mlops/quality.py` | Implemented |
| Prometheus + Grafana | API metrics and dashboard | `api/main.py` (`/metrics`), `docker-compose.monitoring.yml`, `monitoring/prometheus/`, `monitoring/grafana/` | Local stack verified on a bridge network; Streamlit → Grafana Cloud push sent to the real stack (HTTP 200; read-back needs a read token); Azure push implemented, not deployed |
| Evidently 0.7 | Drift detection | `ihds/monitoring.py` (existing), `ihds/mlops/drift.py` | Already existed; also run as a pipeline stage |
| SHAP 0.51 + Fairlearn 0.15 | Explainability and a fairness audit | `ihds/mlops/explain.py` | Implemented |

## Setup

The MLOps tools need their own virtual environment. `prefect` 3.9 requires `fastapi<0.151`, but the app
pins `fastapi==0.165.0`, so the two environments cannot be merged.

```bash
python3.11 -m venv .venv-mlops && source .venv-mlops/bin/activate
pip install -r requirements-mlops.txt
```

Run every command below from the repo root with this environment active.

## The pipeline

```
prepare ─┬─> train (MLflow) ─┬─> evaluate (Deepchecks)
         │        ^          ├─> explain (SHAP + Fairlearn)
validate ┘ (GX gate)         ├─> drift (Evidently)
                             └─> features (Feast)
```

| Run with | Command | What happens |
|---|---|---|
| DVC | `dvc repro` (graph: `dvc dag`, metrics: `dvc metrics show`) | Re-runs only the stages whose deps or params changed. Hashes go to `dvc.lock`; outputs go to the local cache `.dvc/cache`. |
| Prefect | `python -m ihds.mlops.flow` | Runs the same stages as tasks. A failed GX validation or a critical Deepchecks failure raises `QualityGateError`, so the flow ends `Failed` and later tasks do not run. Writes `reports/flow_run.json`. |
| One stage | `python -m ihds.mlops.<prepare\|validate\|train_tracked\|quality\|explain\|drift\|features>` | `validate` and `quality` exit 1 on failure. |

### DVC (data and pipeline versioning)

- `dvc init` has been run. `.dvc/config` disables analytics and sets the default remote `localstore` to the
  folder `dvc-storage/` (gitignored). That remote is a plain local folder, so it needs no credentials.
  `dvc push` copied the cached outputs there.
- `data/pima_diabetes.csv` stays tracked by git because Streamlit and Azure read it directly. DVC versions it
  as a stage dependency: its md5 is recorded in `dvc.lock`, so any change to it re-runs the downstream stages.
  The derived files are DVC outputs and are not in git: `data/processed/*.parquet`,
  `feature_repo/data/patient_features.parquet`, `models/risk_model.joblib`, `reports/deepchecks/`, `reports/drift/`.
- The small JSON metrics in `reports/` and `reports/shap_summary.png` are committed (`cache: false`), so
  reviewers can read the results without running anything.
- `params.yaml` holds the split, validation thresholds, model hyperparameters, the critical Deepchecks checks
  and the age bands.
- Last `dvc repro -f`: 7 of 7 stages ran. After a later edit to `ihds/mlops/validate.py`, `dvc repro` re-ran
  only `validate` and skipped the other 6 stages, because `validation.json` came out unchanged. `dvc status`
  then reported "Data and pipelines are up to date."

### Great Expectations (data validation)

`python -m ihds.mlops.validate` checks the CSV after the same zero-to-missing cleaning the model uses
(GX 1.x API, ephemeral context). The suite has **30 expectations**:

- the column list and its order;
- the row count (700–1000);
- a numeric dtype for each of the 9 columns;
- a non-null share for each column;
- `config.VALID_RANGES` for each of the 8 features (at least 99% of non-null values must fall inside);
- `Outcome` ∈ {0, 1}, with a mean between 0.2 and 0.5.

The non-null thresholds sit just below the observed shares (Insulin is only 51.3% present, SkinThickness
70.4%). They catch a pipeline that loses more data than the source already lacks. They do **not** mean the
data is complete. One BloodPressure value (24 mmHg) is below the guard's 30 mmHg floor, which is why ranges
use `mostly=0.99`.

- **Result: 30 / 30 passed** (`reports/validation.json`; suite definition in `reports/gx_suite.json`).
- A test corrupts 50 Glucose values to 900 and blanks 100 Ages. The suite then fails, naming
  exactly those two expectations.

### MLflow (experiment tracking)

`python -m ihds.mlops.train_tracked` trains the same pipeline as `ihds/model.py`: median impute, then
StandardScaler, then LogisticRegression with `class_weight=balanced`, using hyperparameters from `params.yaml`.

- **What it logs:** params, test-split metrics, 5-fold CV ROC-AUC on the training split,
  `coefficients.json`, and the sklearn model with an input signature and example.
- **Where it's stored:** MLflow 3.x refuses a plain `./mlruns` file store unless `MLFLOW_ALLOW_FILE_STORE=true`
  is set. So metadata goes to the SQLite file `mlruns/mlflow.db` and artifacts go to `mlruns/artifacts/`.
  Both are local and gitignored. Browse them with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Results** (identical in every run; experiment `ihds-risk-model`, 6 runs on the box):
  - test accuracy 0.7338, precision 0.6032, recall 0.7037, F1 0.6496, ROC-AUC 0.8126;
  - confusion matrix TN 75, FP 25, FN 16, TP 38;
  - CV ROC-AUC 0.8444 ± 0.0161.
- **Run IDs:** `dvc repro -f` produced `8b95d9818eae4b4d94177879cdde1f3c` (stored in `reports/train_metrics.json`).
  The last Prefect flow produced `b1085973ece0449c93940c9b9d94725c` (stored in `reports/flow_run.json`).
- **Check:** a test reloads the logged model from MLflow and confirms it gives the same probabilities as the
  in-memory model.

### Deepchecks (model-quality testing)

`python -m ihds.mlops.quality` runs three built-in suites: `data_integrity` on the full cleaned data,
`train_test_validation`, and `model_evaluation` with the trained model. It writes the HTML reports to
`reports/deepchecks/` (about 4.8 MB each, gitignored) and a summary to `reports/deepchecks_summary.json`.

**The gate:** the stage exits 1 if any of these checks fails, or if test ROC-AUC is below 0.75:

- Data Duplicates, Mixed Nulls, Conflicting Labels;
- Train Test Samples Mix, New Label Train Test;
- Label / Feature / Prediction Drift;
- Train Test Performance.

| Suite | Checks | Pass | Warn | Fail | Not run* |
|---|---|---|---|---|---|
| data_integrity | 12 | 11 | 0 | 0 | 1 |
| train_test_validation | 12 | 8 | 0 | 1 — Feature Label Correlation Change | 3 |
| model_evaluation | 18 | 12 | 2 — Weak Segments (train/test) | 1 — Simple Model Comparison | 3 |

\*These checks cannot apply to this data: there is no datetime or index column, and some checks are for
regression or boosting models only.

**Result: gate passed. No critical failures; test ROC-AUC 0.8126.** Two checks failed without being critical:

- **Feature Label Correlation Change:** Insulin's predictive power score differs by 0.3 between train and
  test. Insulin is about 49% missing, and the test split is small (n = 154).
- **Simple Model Comparison:** for class 0 the F1 gain over Deepchecks' simple baseline is −0.97%, below the
  10% condition. The balanced class weights trade class-0 F1 for recall on the diabetic class.

These two are reported, not hidden. A test feeds a "test" set containing 94 training rows. The gate then
fails on Train Test Samples Mix.

**Compatibility note:** deepchecks 0.19.1 is the latest release and still uses NumPy aliases that NumPy 2
removed (`np.Inf`, …). `quality._numpy2_aliases()` re-adds those aliases before the import. It only adds
names; it changes no values.

### SHAP and Fairlearn (explainability and fairness)

`python -m ihds.mlops.explain` uses `shap.LinearExplainer` on the logistic-regression step, over the imputed
and standardized features, with a background of 100 training rows. Values are in log-odds.

**Global ranking** (mean |SHAP| over the 154 test patients, in `reports/shap_global.json`, plotted in
`reports/shap_summary.png`):

| Feature | mean \|SHAP\| |
|---|---|
| Glucose | 1.0136 |
| BMI | 0.5870 |
| Pregnancies | 0.3285 |
| DiabetesPedigreeFunction | 0.2051 |
| Age | 0.1488 |
| Insulin | 0.0297 |
| BloodPressure | 0.0106 |
| SkinThickness | 0.0088 |

Local explanations for the first 3 test patients list their top 3 features. A test checks additivity: base
value plus the sum of SHAP values equals the model's log-odds.

**Fairness: the attribute used.** The Pima data has no sex, race or ethnicity column; every row is a Pima
Indian woman aged 21 or over. So the only attribute that can be audited is **age**, in the bands 21–29,
30–44 and 45+. Fairlearn `MetricFrame` ran on the test split at threshold 0.5 (`reports/fairness.json`):

| Age band | n | Diabetic | Accuracy | Recall (TPR) | Precision | Selection rate | FPR |
|---|---|---|---|---|---|---|---|
| 21–29 | 85 | 16 | 0.741 | 0.438 | 0.350 | 0.235 | 0.188 |
| 30–44 | 40 | 26 | 0.700 | 0.731 | 0.792 | 0.600 | 0.357 |
| 45+ | 29 | 12 | 0.759 | 1.000 | 0.632 | 0.655 | 0.412 |

**Disparity results:**

- Demographic parity difference **0.420** (ratio 0.359).
- Equalized odds difference **0.563** (ratio 0.438).

These are large gaps. Age is itself a model input and a real clinical risk factor, so a higher selection rate
in older bands is expected. The finding that matters is that **recall is lowest for 21–29-year-olds
(7 of 16 diabetic patients flagged)**.

The bands are small, so these estimates are noisy. The audit is reported for review and does not fail the
pipeline. No mitigation (for example, per-group thresholds) has been applied.

### Evidently (drift) as a pipeline stage

`python -m ihds.mlops.drift` reuses `ihds.monitoring.run_all` and writes to `reports/drift/`. The Streamlit
Monitoring tab keeps using the committed `monitoring/reports/`, which this stage does not touch.

- **Holdout:** no drifted features; accuracy 0.7338.
- **Shifted** (simulated +25 mg/dL glucose meter error): Glucose flagged (1 of 8 columns); precision 0.6032 → 0.5269,
  recall 0.7037 → 0.9074, accuracy 0.6818.

### Feast (feature store)

- **Definitions:** `feature_repo/features.py` defines entity `patient` (join key `patient_id`), a `FileSource`
  over `feature_repo/data/patient_features.parquet`, the feature view `patient_clinical_features` (8 features
  plus `Outcome`) and the feature service `risk_model_v1`.
- **Storage:** `feature_store.yaml` uses the local provider, with SQLite for both the registry and the online
  store, all under `feature_repo/data/` (gitignored).
- **Honest limits:** `patient_id` is just the CSV row number, and every row shares one fixed
  `event_timestamp` (2026-01-01). The dataset has neither real IDs nor timestamps.
- **Commands:**

```bash
python -m ihds.mlops.features --patient-id 0 --patient-id 5   # apply + materialize + online fetch
python -m ihds.mlops.features --patient-id 9 --no-apply       # fetch only
# CLI equivalent, from feature_repo/:
feast apply && feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
```

**Result:**

- `feast apply` created the project `ihds`, entity `patient`, feature view `patient_clinical_features`,
  feature service `risk_model_v1` and the SQLite table.
- The online fetch for patient 0 returned Glucose 148, BMI 33.6, Age 50, BloodPressure 72, SkinThickness 35,
  Pregnancies 6, DPF 0.627, and Insulin `None` (the raw 0, read as missing).
- The fetched values match the source parquet (`matches_source_parquet: true`).
- Scored on those fetched features, the model gives p = 0.826 for patient 0 and 0.2059 for patient 5
  (`reports/feast_online_sample.json`).

### Prefect (orchestration)

`python -m ihds.mlops.flow` ran the flow `ihds-mlops-pipeline` on a temporary local Prefect API; no Prefect
Cloud and no `PREFECT_API_URL` were needed.

**Result:** all 7 tasks reached `Completed` (prepare, validate-gx, train-mlflow, evaluate-deepchecks,
explain-shap-fairlearn, drift-evidently, feature-store-feast), and the flow state was **Completed**. It took
37 s wall time, including the temporary server's start-up.

A test uses a throwaway Prefect API (`prefect_test_harness`) and checks two things:

- the task order;
- that a failing GX result stops the flow at `validate`, leaving it `Failed`.

## Prometheus and Grafana

**The endpoint:** `api/main.py` serves `GET /metrics` through `prometheus_client`. Route labels use the route
template, and `/metrics` itself is not counted. The series are:

- `ihds_http_requests_total{method,route,status}`
- `ihds_http_request_duration_seconds` (histogram)
- `ihds_predictions_total{band}` (low / moderate / high)
- `ihds_assess_total{status,draft_source}`
- `ihds_escalations_total`
- `ihds_reviewer_revisions_total`
- `ihds_assess_duration_seconds` (workflow duration histogram)
- `ihds_app_up`, `ihds_app_start_time_seconds`

The Streamlit Cloud push (below) uses the same names.

The dashboard `monitoring/grafana/dashboards/ihds_api.json` has 10 panels:

- stat panels for assessments, escalations, high-risk predictions and blocked. Each one counts events in the
  selected time range with `round(sum(increase(...[$__range])))`, which survives counter resets when a process
  restarts. Dashboard version 4: re-import the JSON if you imported an earlier version;
- API request rate per route, split by `service` (the Docker/Azure API and the API embedded in the Streamlit Cloud
  app);
- assessment latency p50 / p95;
- predictions per band;
- outcomes and escalations;
- draft source and reviewer revisions;
- push heartbeat.

Its datasource is a dashboard variable (`${datasource}`, type Prometheus), not a hard-coded uid. A `service`
variable selects the API (`ihds`), Streamlit (`ihds-streamlit`) or both. The same file therefore works with the
provisioned local datasource and can be imported into Grafana Cloud.

The 104-call results below were measured before the latency panel moved to `ihds_assess_duration_seconds`.

### Local stack (`docker-compose.monitoring.yml`)

```bash
docker compose -f docker-compose.monitoring.yml up -d --build --wait
# API http://127.0.0.1:8000  Prometheus http://127.0.0.1:9090  Grafana http://127.0.0.1:3000 (anonymous viewer)
docker compose -f docker-compose.monitoring.yml down -v
```

- **Services:** `api` (the app image, FastAPI only, on 0.0.0.0:8000 inside the container), `prom/prometheus:v3.5.0`
  and `grafana/grafana:12.1.0`.
- **Network:** all three are on one user-defined bridge network, `ihds-monitoring`. Prometheus scrapes
  `api:8000/metrics` by service name every 15 s.
- **Grafana:** gets its datasource URL from `PROMETHEUS_URL=http://prometheus:9090`. It provisions the datasource
  (uid `ihds-prometheus`) and the dashboard.
- **Startup order:** each service has a healthcheck: API `/health`, Prometheus `/-/ready`, Grafana `/api/health`.
  `depends_on: service_healthy` starts them in the order api → prometheus → grafana, and `--wait` returns only
  when all three are healthy.
- **Ports:** all bind to 127.0.0.1. Grafana's admin login keeps its default, so the stack is for local use only.

### Why bridge networking failed on the box, and the fix

**Root cause:** the box kernel has two sets of firewall tables, and it evaluates both:

- **iptables-nft:** what dockerd uses. Docker adds its `br-*` ACCEPT rules here.
- **iptables-legacy:** stale tables. Their `FORWARD` chain has policy `DROP` and rules only for `docker0`.

So traffic between containers worked on the default `docker0` bridge but was dropped on every user-defined
`br-*` network, which is what Compose creates.

Evidence:

- The legacy `FORWARD` DROP counter rose while the scrape timed out (158 → 233 packets).
- `iptables-legacy -S | grep br-` returned no rules.
- With the unmodified compose file, Prometheus showed `ihds-api down: Get "http://api:8000/metrics": context
  deadline exceeded`.

**The fix:** this is a box problem, not a compose problem. `scripts/fix_docker_bridge_forward.sh` adds two rules
to the legacy `DOCKER-USER` chain:

- `-i br-+ -j ACCEPT`
- `-o br-+ ... RELATED,ESTABLISHED -j ACCEPT`

It is idempotent and does nothing unless the legacy `FORWARD` policy is `DROP`. The rules do not survive a
reboot. A normal Docker host needs none of this.

**Compose-file fixes:**

- explicit network;
- scrape by service name;
- API bound to 0.0.0.0;
- datasource URL from env;
- healthchecks and `depends_on`.

The earlier host-network override was removed.

### Verified on the box (2026-11-25), unmodified compose file on a bridge network

- **Network:** `docker network inspect` showed all three containers on `ihds-monitoring_ihds-monitoring`.
- **Targets:** after the fix script, with the compose file unchanged, Prometheus showed `ihds-api up
  http://api:8000/metrics`.
- **Grafana:** the datasource URL was `http://prometheus:9090`, and health returned "Successfully queried the
  Prometheus API".
- **Traffic:** a script sent 104 `POST /assess` calls (the first 100 Pima rows plus 4 dosing questions).
  Prometheus then returned:
  - `ihds_assess_total`: ok 95, blocked 9;
  - `ihds_predictions_total`: high 35, low 33, moderate 27;
  - `ihds_escalations_total` 0 (template writer);
  - p95 latency 0.0486 s.
- **Panels:** every dashboard panel returned series through Grafana `/api/ds/query`.
- **Screenshot:** `docs/screenshots/10_grafana_dashboard.png`, taken from this run.
- **CI rehearsal:** the `monitoring-stack` job's commands from `docs/ci.yml.example` were run by hand. They
  passed:
  - `compose config` and `promtool check config`;
  - `up --wait`, with all 3 healthy;
  - targets `{'ihds-api': 'up', 'prometheus': 'up'}`;
  - `sum(ihds_predictions_total)` = 20 after 20 calls;
  - Grafana `/api/health`, datasource health OK, and the dashboard found by search.

### Pushing metrics from Azure Container Apps

Container Apps exposes only Streamlit (8501), so nothing outside the container can scrape `/metrics`. Instead,
the app image includes a Prometheus binary (`COPY --from=prom/prometheus:v3.5.0`, about 160 MB).

**How the push works:**

- When `GRAFANA_RW_URL` is set, `scripts/start_all.sh` starts `scripts/metrics_agent.py`.
- `metrics_agent.py` runs `prometheus --agent`, which scrapes `127.0.0.1:8000/metrics` inside the container
  every 15 s.
- It remote-writes only `ihds_*` and `up` to that URL. It drops `*_created` and all python/process series.
- It adds the labels `service="ihds"` and `deployment="azure-container-apps"`.

**When it is off:** with `GRAFANA_RW_URL` unset (the default), no agent starts, and the image behaves as before.

**Token handling:**

- `GRAFANA_RW_TOKEN` is written to a 0600 file and used through `basic_auth.password_file`.
- The token is never written to the config, and the agent process is started without it.
- It is also removed from the environments of the MCP server, uvicorn and Streamlit.
- Grafana Alloy was considered and rejected: its binary is 370 MB, against 159 MB for Prometheus.

**Verified locally** (`deploy/monitoring/docker-compose.push-test.yml`, the same image with a local Prometheus
receiver standing in for Grafana Cloud):

- The log showed `metrics agent: pushing ihds_* series to ...`.
- The token was not in `agent.yml`, the token file was mode 600, and the agent's environment had no
  `GRAFANA_RW_TOKEN` or `OPENAI_API_KEY`. The uvicorn and Streamlit environments had no token either.
- After 31 calls, the receiver had `ihds_predictions_total{band="high",service="ihds",deployment="local-push-test"}`
  = 30, and `ihds_assess_total` ok 30, blocked 1.
- Pushed metric names were only `ihds_*` and `up`. No `python_*` or `process_*` series arrived, and no new
  `*_created` samples after the drop rule.
- Grafana (the option B image) showed data in all 8 panels of the dashboard version at that time.
- **Agent off:** with no `GRAFANA_RW_URL`, the image started MCP, uvicorn and Streamlit only, with no agent
  process and no `/tmp/ihds-agent`.
- **Grafana Cloud import:** importing `ihds_api.json` through Grafana 12.1's `/api/dashboards/import` into a
  Grafana whose Prometheus datasource has a different uid succeeded. That version of the dashboard had 8 panels, all bound to
  `${datasource}`.

**Not verified:** a push to a real Grafana Cloud stack, and any Azure deployment. Neither has been run.

#### Option A: Grafana Cloud (recommended)

1. **Get the credentials.** In Grafana Cloud, open **My Account → your stack → Prometheus → Details**. Copy:
   - the **Remote Write Endpoint** (it ends in `/api/prom/push`);
   - the **Username / Instance ID** (a number).

   Under *Sending metrics*, click **Generate now** to create an access-policy token with `metrics:write`
   (it starts with `glc_`). It is shown once. Do not paste it into chat or commit it.
2. **Turn on the push.** In Azure Cloud Shell (Bash), from the repo root (`git pull` first):

   ```bash
   export RG=Regroup_9vxmyzpemb
   export ACR=$(az acr list -g "$RG" --query "[0].name" -o tsv); echo "$ACR"
   export TAG=v3                                   # a NEW tag, so Container Apps starts a new revision
   export GRAFANA_RW_URL='https://prometheus-prod-XX-prod-REGION.grafana.net/api/prom/push'
   export GRAFANA_RW_USER='<instance ID>'
   read -r -s -p "Grafana Cloud token: " GRAFANA_RW_TOKEN; echo; export GRAFANA_RW_TOKEN
   bash deploy/azure/enable_metrics_push.sh        # secret set + az acr build ihds:$TAG + containerapp update
   unset GRAFANA_RW_TOKEN
   ```

   The script stores the three values as Container Apps secrets: `grafana-rw-url`, `grafana-rw-user` and
   `grafana-rw-token`. It maps them to env vars with `secretref:`. The same steps by hand:

   ```bash
   az containerapp secret set -n ihds-app -g "$RG" --secrets grafana-rw-url="$GRAFANA_RW_URL" \
     grafana-rw-user="$GRAFANA_RW_USER" grafana-rw-token="$GRAFANA_RW_TOKEN"
   az acr build -r "$ACR" -t "ihds:$TAG" .
   az containerapp update -n ihds-app -g "$RG" --image "$ACR.azurecr.io/ihds:$TAG" \
     --set-env-vars GRAFANA_RW_URL=secretref:grafana-rw-url GRAFANA_RW_USER=secretref:grafana-rw-user \
     GRAFANA_RW_TOKEN=secretref:grafana-rw-token
   ```

3. **Check the push.**
   - Run `az containerapp logs show -n ihds-app -g "$RG" --tail 80 | grep "metrics agent"`.
   - Run a few assessments in the app UI.
   - In Grafana Cloud **Explore**, query `ihds_assess_total{service="ihds"}`.
4. **Import the dashboard.**
   - Go to **Dashboards → New → Import → Upload dashboard JSON file**, pick
     `monitoring/grafana/dashboards/ihds_api.json`, then **Import**.
   - In the dashboard's **Data source** dropdown, pick `grafanacloud-<stack>-prom` (usually the default), then
     save.
5. **Maintenance.**
   - **Rotate the token:** run `BUILD=0` with the new token. The script restarts the active revision.
   - **Turn it off:** `RG=... DISABLE=1 bash deploy/azure/enable_metrics_push.sh`.
   - **Free tier:** 10k active series and 14-day retention. IHDS pushes a few dozen series.

#### Option B: self-hosted Prometheus + Grafana on Container Apps (lower priority)

`deploy/azure/deploy_monitoring_aca.sh` builds two images in ACR:

- `deploy/monitoring/prometheus/Dockerfile`: a remote-write receiver with 15-day retention;
- `deploy/monitoring/grafana/Dockerfile`: the provisioning and dashboard baked in, anonymous Viewer.

It then creates two apps in `ihds-env`:

- `ihds-prometheus`: internal ingress only, on 9090;
- `ihds-grafana`: external ingress on 3000, with the admin password as a secret.

Finally, it calls `enable_metrics_push.sh` with `GRAFANA_RW_URL=http://ihds-prometheus/api/v1/write` (internal,
no credentials).

```bash
export RG=Regroup_9vxmyzpemb ACR=<acr> TAG=v3
bash deploy/azure/deploy_monitoring_aca.sh    # prompts for a Grafana admin password (hidden, 12+ chars)
```

**Caveats:**

- Prometheus data is lost on every new revision, since there is no volume.
- The receiver has no auth, but it is internal to the environment.
- The two extra apps use lab credit.
- `az acr build` may be blocked by lab policy.
- The admin password applies only on Grafana's first start.

**Verified:** both images were built and run locally (push-test above). They have not been deployed to Azure.

### Streamlit Community Cloud → Grafana Cloud (direct push)

The Streamlit Cloud app (https://ihds-diabetes-advisor.streamlit.app) runs the FastAPI app inside its own process
(see "Embedded REST API" below), with tools over `mcp-stdio`. It cannot run a Prometheus binary or expose
`/metrics`, so `ihds/metrics_push.py` pushes from the Streamlit process itself.

**How the push works:**

- **Protocol:** Prometheus remote write 1.0 to Grafana Cloud's `/api/prom/push`, with HTTP basic auth (instance
  ID and an access-policy token with `metrics:write`).
- **No new dependencies:**
  - The protobuf `WriteRequest` is three tiny messages, encoded by hand.
  - The snappy block uses literal-only encoding, which is valid snappy that any decoder accepts.
  - HTTP uses `urllib`.
  - `requirements.txt` is unchanged.
  - The encoder's bytes are identical to `protobuf`'s own serialization, and its snappy output decodes with
    `cramjam` (both checked on the box).
  - Influx line protocol and OTLP were not used: they rename or retype metrics, or need an SDK.
- **Delivery:** a daemon thread pushes every 15 s (`IHDS_PUSH_INTERVAL`) with a 5 s timeout.
- **Failure handling:** errors are counted and logged without credentials, and retried on the next tick. They
  never raise into the app. `record_assessment` is a no-op when the push is off.
- **Off by default:** nothing starts unless `GRAFANA_RW_URL` is configured, either in `st.secrets` (top level or
  a `[grafana]` table) or in the environment.
- **Token handling:**
  - The token stays inside the module. It is not copied to `os.environ`, and the MCP stdio server starts with
    the MCP SDK's minimal default environment.
  - Use the `[grafana]` table: Streamlit exports only root-level secrets as environment variables.
  - Checked on the box: neither the Streamlit nor the MCP process had any `GRAFANA_RW_*` variable.
- **Sidebar:** shows `Metrics push: on → <host> · N ok / M failed` or `Metrics push: off`.
- **API mode:** when the app calls an external REST API (`IHDS_API_URL`, Docker/Azure), Streamlit neither
  counts nor pushes, because that API already exposes `/metrics`.
- **Embedded API:** the pusher also sends the in-process FastAPI app's `ihds_*` series from `prometheus_client`'s
  default registry (`include_api_metrics`), relabelled `service="ihds-streamlit"`: `ihds_http_requests_total`,
  `ihds_http_request_duration_seconds` and the assessment counters. `ihds_app_*` and `*_created` are skipped (the
  pusher sends its own heartbeat). If a label set exists in both registries (the latency histogram), the API's
  wins, so each series is sent once. The sidebar adds ` · incl. API metrics`.

**Series:** the names are shared with `api/main.py`, so one dashboard covers both:

- `ihds_assess_total{status,draft_source}`
- `ihds_predictions_total{band}`
- `ihds_escalations_total`
- `ihds_reviewer_revisions_total`
- `ihds_assess_duration_seconds` (histogram, 0.05–60 s)
- `ihds_app_up` (heartbeat)
- `ihds_app_start_time_seconds`

New counter series are pushed with an explicit 0 sample one push interval before their first value. Without
it, `increase()` would miss the first event of a new label combination. The API pre-creates its common label
combinations at 0 for the same reason. Checked on a local receiver: 5 assessments across a simulated process
restart gave stat panels of 5 / 0 / 3 / 1. The old raw-counter expression showed 3.

Earlier test pushes to the real stack are labelled `deployment="local-box"`, `local-box-server`, or
`service="ihds-pytest"` / `ihds-push-probe`. They count only inside a time range that includes them. Select
`service=ihds-streamlit` and a range after 2026-11-25 08:00 UTC to exclude them.

Labels: `service="ihds-streamlit"`, `source="streamlit"`, `job="ihds-streamlit"`,
`deployment=IHDS_METRICS_DEPLOYMENT` (default `streamlit-cloud`) and `instance=<hostname>`. Counters start at 0
when the process starts. `increase()` and `rate()` handle restarts, and the stat panels use `increase()` over the dashboard range.

**Dashboard:** `monitoring/grafana/dashboards/ihds_api.json`, now "IHDS (API + Streamlit)", has 10 panels and two
variables:

- `datasource`: type Prometheus;
- `service`: `label_values(ihds_assess_total, service)`, multi-select, with All. The local Prometheus and the
  Container Apps agent label the API `service="ihds"`.

The latency panel now uses `ihds_assess_duration_seconds`, which the API also records. Two panels are new:

- draft source and reviewer revisions;
- push heartbeat: seconds since the newest `ihds_app_up` sample.

**Streamlit Cloud setup:**

1. Go to share.streamlit.io, find **ihds-diabetes-advisor**, then choose **⋮ → Settings → Secrets**.
2. Add the following, keeping any existing `OPENAI_*` lines, then **Save**:

   ```toml
   [grafana]
   GRAFANA_RW_URL = "https://prometheus-prod-XX-prod-REGION.grafana.net/api/prom/push"
   GRAFANA_RW_USER = "<Grafana Cloud Prometheus instance ID>"
   GRAFANA_RW_TOKEN = "glc_<paste the metrics:write token here>"
   ```

3. Streamlit restarts the app. Open it: the sidebar should say `Metrics push: on`.
4. In Grafana Cloud, go to **Dashboards → New → Import → Upload dashboard JSON file**, choose `ihds_api.json`,
   then **Import**. Pick `grafanacloud-<stack>-prom` in **Data source**, then save.

**Verified on the box, 2026-11-25, against the real Grafana Cloud stack:**

- **AppTest run:** `app.py` ran through `streamlit.testing.v1.AppTest` with the credentials in `st.secrets`, and
  12 assessments were run (11 ok, 1 blocked dosing question; high 5, low 4, moderate 2). The pusher made 5
  pushes, all HTTP 200, 0 failed.
- **Server run:** `streamlit run` with the credentials in a user-level `secrets.toml` outside the repo, and 3
  button clicks in headless Chrome. The log showed `metrics push: sending ihds_* series to https://…/api/prom/push`,
  with 0 push failures.
- **Endpoint checks:** the same endpoint rejected bad input synchronously, so a 200 means the payload was decoded
  and validated:
  - a wrong token gave 401;
  - a non-snappy body gave 400 `snappy: corrupt input`;
  - an invalid metric name gave 400 `err-mimir-metric-name-invalid`;
  - a timestamp 2 h ahead gave 400 `err-mimir-too-far-in-future`.
- **Not verified: reading the data back from Grafana Cloud.** The token is write-only: `/api/prom/api/v1/query`
  returned 401 `invalid scope requested`. To confirm, query `ihds_assess_total{service="ihds-streamlit"}` in
  Grafana Cloud **Explore**, or use a token with `metrics:read`.
- **Dashboard (local):** imported into a local Grafana 12.1 whose datasource has another uid. It was fed by
  `metrics_push` through a local Prometheus remote-write receiver (40 synthetic assessments). The `service`
  variable resolved to `ihds-streamlit`, and 9 of 10 panels returned data. The API request-rate panel was empty
  then, because Streamlit had no API; it gets data since the embedded API (dashboard version 4). Latency p50 0.03 s, p95 0.085 s; heartbeat 0.2 s.
- **Dashboard (Grafana Cloud):** not imported through the API. That needs the stack URL and a Grafana service
  account token, and the write token cannot list stacks.

### Embedded REST API in the Streamlit Cloud app

Streamlit Community Cloud runs only `streamlit run app.py`, so `ihds/embedded_api.py` starts the FastAPI app
(`api/main.py`) inside the Streamlit process. The UI then uses the same `POST /assess` route as Docker and Azure.

- **Start:** once per process (`st.cache_resource`), so Streamlit reruns do not start it again. `uvicorn.Server`
  runs in a daemon thread on `127.0.0.1` and a free port, with no signal handlers and no access log. The app waits
  for `GET /health` (30 s timeout, polled every 0.2 s).
- **Shared state:** the API reuses the app's `System` (`api.main.STATE["system"]`): the same model, the same
  Chroma index and one MCP stdio child. There is no second training run and no second MCP process. The MCP HTTP
  server is not started (it is optional); tools stay `mcp-stdio`.
- **Gate:** `IHDS_EMBED_API`, default on. It is off when `IHDS_API_URL` is set, and `scripts/start_all.sh` sets
  `IHDS_EMBED_API=0` for Docker/Azure, where the API runs as its own process.
- **Fallback:** if the start or a call fails, the app runs the graph in-process. The sidebar then shows
  `Embedded REST API unavailable, using in-process: <reason>`, or a warning for one failed call. That
  assessment is counted in the API's metrics, so it is still pushed.
- **Sidebar and trace:** `Workflow via REST API http://127.0.0.1:<port> · tools: mcp-stdio`, `REST API embedded
  in this process (started in X s)`, and `Workflow route: REST API ... (embedded)`.
- **Tracing:** LangSmith tracing, the Chroma retrieval and the Grafana push are unchanged. Tokens stay out of the
  MCP child: on the box its environment had only HOME, LOGNAME, PATH, TERM, USER, PYTHONPATH and IHDS_GAN_STEPS.
- **Startup:** about 0.2 s on the box, because the `System` is already built. The first page load still pays for
  the model and Chroma build (and the ONNX download on Streamlit Cloud).
- **Code updates (git push):** Streamlit Cloud applies a push by rerunning `app.py` and reloading only the
  modules that changed, while `st.cache_resource` keeps the System and the embedded API built from the old
  code.
  - Seen on 2026-11-25 after commit 0919e9c: the page showed the new "View trace in LangSmith" link, but the
    API on the same port still ran the old reviewer (grounding 0.185, rejected three times). The sidebar had
    also lost "incl. API metrics".
  - `app.py` now hashes `ihds/`, `api/` and `mcp_server/`. When the hash changes, it drops every project module
    except the metrics pusher and imports them again.
  - It then rebuilds the System, the LLM graph and the embedded API (the caches are keyed by the hash, with
    `max_entries=1`). It stops the old uvicorn server and closes the old MCP stdio child.
  - It re-enables the API-metrics push on every rerun. `api/main.py` reuses its Prometheus collectors when
    imported again.
  - Checked locally: after a source change and a page reload, the API moved to a new port, the old port was
    closed, and one MCP child remained. The push went on every 5 s with one pusher and "incl. API metrics".
  - The first page load after a push retrains the model (as on a cold start).
- **Caveats:**
  - The API has no authentication. It listens only on 127.0.0.1 inside the app container.
  - `/metrics` is not reachable from outside; the push is the only way out.
  - All sessions share one API and one MCP child, as before with the shared `System`.
  - If Streamlit Cloud ever runs several processes, each starts its own API on its own port, with its own
    `instance` label.

**Checked on the box** (`streamlit run`, headless Chrome, a local remote-write receiver):

- The sidebar showed `Workflow via REST API http://127.0.0.1:46453 · tools: mcp-stdio` and `started in 0.2 s`.
- One assessment returned p=0.60, high, approved, with retrieval `chroma` and a LangSmith run link.
- The receiver got `ihds_http_requests_total{service="ihds-streamlit",method="POST",route="/assess",status="200"}`
  with a 0 sample first, then 1. No push sent duplicate series.
- Only one MCP child ran.

**Checked live on Streamlit Cloud** (2026-11-25 ~01:25 PT, commit f62f13c, headless Chrome):

- The sidebar showed `Workflow via REST API http://127.0.0.1:34797 · tools: mcp-stdio`, `REST API embedded in this
  process (started in 0.2 s)`, `LLM key configured on the API server` and `Metrics push: on → <stack host> · 16 ok /
  0 failed (last 200) · incl. API metrics` (the count had been 2 when the page first loaded).
- One gpt-4o-mini assessment returned 60 %, high, approved, with draft source `llm` and 1 reviewer revision.
- The trace showed `Workflow route: REST API http://127.0.0.1:34797 (embedded) · retrieval: chroma` and
  `evidence_agent: [retrieval: chroma (onnx-all-MiniLM-L6-v2)]`.
- Not seen in Grafana Cloud: the write-only token cannot read data back, so look for
  `ihds_http_requests_total{service="ihds-streamlit"}` in Explore.
- LangSmith showed `off (no LANGSMITH_API_KEY)` because the Cloud app has no `[langsmith]` secret yet.

Tests: `tests/test_embedded_api.py` covers the gate, `/health` and `/assess` on the shared `System`, the pushed
API series (labels, zero-first, no duplicates, no `_created`), and the Streamlit app routed through the embedded
API (AppTest).

## Evidence retrieval: ChromaDB with a keyword fallback

`rag_search` (used by `evidence_agent`) now retrieves from a ChromaDB index of the 14 evidence cards (K1–K14
in `data/clinical_knowledge.csv`). The original TF-IDF keyword matcher is the fallback.

**Code:** `ihds/workflow.py` (`KnowledgeBase.retrieve`) and `ihds/vector_store.py`.

**Settings:**

- **`IHDS_RETRIEVAL`:** `auto` (default) or `chroma` uses Chroma; `keyword` uses TF-IDF only.
- **Fallback:** if `chromadb` is missing, the model cannot load, or a query raises an error, the query is
  answered by keywords and the reason is kept.
- **Embeddings (`IHDS_EMBEDDINGS=onnx`, the default):** chromadb's bundled all-MiniLM-L6-v2 model, run locally
  with onnxruntime. It needs no key; the ~80 MB model is downloaded once to `~/.cache/chroma` (167 MB on disk).
  The Docker image bakes it in.
- **`IHDS_EMBEDDINGS=openai`:** `text-embedding-3-small` through Vocareum. It was tested and works (1536
  dimensions), but only in a process that has `OPENAI_API_KEY`. The MCP tool server, where `rag_search` runs,
  deliberately has no key, so there it falls back to ONNX. It is used in the comparison below.
- **Client:** in-memory `EphemeralClient` by default. Set `IHDS_CHROMA_DIR=.chroma` (gitignored) for a
  `PersistentClient`. The collection name includes a hash of the cards and the model.
- **Cache:** query results are cached, because workflow queries repeat. An uncached MiniLM query takes about
  0.13 s on the box.

**Where it shows:**

- The tool returns `{"evidence", "method", "embedding"}`.
- The trace shows `evidence_agent: [retrieval: chroma (onnx-all-MiniLM-L6-v2)] ['K4', 'K2', 'K1']; ...`.
- `AdvisoryResult.retrieval` and the API response carry `chroma` or `keyword`.
- The UI's evidence and trace panels show the method.

**Install cost:** chromadb 1.5.9 plus onnxruntime 1.32.0 add 33 packages and about 281 MB installed on Python
3.14 (849 → 1130 MB). The pinned `requirements.txt` resolves on 3.11–3.14 from binary wheels, and `uv pip check`
reports no conflicts on 3.14. The Docker image (Python 3.11) builds, and its container answered `/assess` with
`retrieval: chroma` over `mcp-http`, with no model download at run time.

**Comparison, `python scripts/compare_retrieval.py --llm 30 --repeats 2`:** results are in
`reports/retrieval_comparison.json`, with top-3 for 11 queries per method.

- **Overlap@3:** keyword vs Chroma-ONNX 0.788; keyword vs Chroma-OpenAI 0.758; ONNX vs OpenAI 0.818.
- **Top-1 agreement:** 0.636 / 0.545 / 0.727.
- **"Should this patient be screened again?":** keyword returns nothing; Chroma returns K5, K11, K12.
- **"Is there a family history component?":** all methods put K6 first.
- **Glucose + BMI query:** keyword gives K4, K2, K1; ONNX gives K2, K1, K4; OpenAI gives K2, K1, K3.

**Live gpt-4o-mini review:** 3 runs × 30 held-out cases per retriever, in-process tools.

| | Approved first try (each run) | Mean | Final approval | Escalated (of 90) | Mean grounding |
|---|---|---|---|---|---|
| keyword | 0.733 / 0.733 / 0.667 | 0.711 | 1.0 | 13 | 0.33–0.35 |
| chroma (ONNX) | 0.600 / 0.733 / 0.733 | 0.689 | 1.0 | 11 | 0.29–0.31 |

The approval rate did not collapse: the difference is within the run-to-run spread. Mean grounding is about
0.04 lower with Chroma, and every run stayed at or above the reviewer's 0.2 threshold (minimum 0.2). The committed
`monitoring/reports/llm_eval_summary.json` (keyword era, first try 0.767) was not regenerated.

## Reviewer grounding: root cause and fix (2026-11-25)

**Symptom:** on Streamlit Cloud with Chroma retrieval, the reviewer rejected all three gpt-4o-mini drafts of the
default case on `grounding_ok` (0.182 < 0.2). The template was released and the case escalated.

**Root cause:** the score compared against the wrong sources, and Chroma was not the cause.

- `review()` scored a draft only against the evidence cards. It measured the share of the draft's content words
  found in the retrieved `fact` texts.
- The writer is told to state the probability, the band and the drivers, and it is given the population
  context. The reviewer requires the disclaimer. None of those words are in the cards.
- In the real default-case draft that scored 0.185, 44 of 54 content words were "ungrounded". Most of them came
  from the case data (`probability, band, high, drivers, log, odds, age, years, mean, diabetic, population,
  welch, 110, 142`) or the disclaimer (`advisory, draft, diagnosis, clinician, review, sign, off, required`).
- A faithful draft therefore sat at about 0.18 to 0.22, and approval was close to a coin flip. With temperature
  0, the retries came back almost the same, because the writer's feedback was only the check name
  (`grounding_ok`).
- Keyword and Chroma return the same cards for the default case (K7, K2, K1). In the 15-case batch below, the
  two retrievers fail at similar rates.
- The prompt also put the case data and the cards in one block. The model then cited a card for case data:
  the population means were tagged `[K2]`.

**Change** (`ihds/workflow.py`, `ihds/agents.py`):

- **Sources:** the reviewer now grounds a draft against everything the writer was given. That is the evidence
  cards plus the case-data lines (`agents.case_facts`: probability and band, drivers with direction and
  log-odds, population context, robustness check, imputed fields).
- **Disclaimer:** the mandatory disclaimer is removed before scoring. `has_disclaimer` checks it on its own.
- **Recalibrated threshold:** `MIN_GROUNDING` went from 0.2 to **0.4**. The threshold was raised, not lowered, so
  invented content still fails. On the default case:
  - the faithful 0.185 draft scores 0.578;
  - drafts with invented clinical claims or numbers (in `tests/test_workflow.py`) score 0.14 to 0.32 and are
    rejected;
  - the template scores 1.0.
- **Feedback:** the reviewer returns actionable feedback (`review["feedback"]`), for example the words not found
  in any source and the card ids that may be cited. It also returns `unsupported_terms`. The writer receives
  this feedback instead of check names.
- **Prompt:** the writer prompt separates "Case data" from "Evidence cards". It says to keep to their wording,
  cite `[K#]` only for statements taken from a card, never cite a card for case data, and add no other facts.

**Measured:** `scripts/eval_grounding.py`, real gpt-4o-mini through Vocareum, in-process tools. The 15 cases are
the default case and the first 14 held-out cases with glucose and BMI. Results are in
`reports/grounding_eval.json`, including every draft.

| Code | Retriever | First-try approval | LLM draft released | Escalated (of 15) | LLM calls | Mean grounding (reviewer's score) | Mean grounding, evidence cards only |
|---|---|---|---|---|---|---|---|
| before | keyword | 0.600 | 0.800 | 3 | 25 | 0.218 | 0.218 |
| before | chroma | 0.667 | 0.933 | 1 | 22 | 0.207 | 0.207 |
| after | keyword | 1.000 | 1.000 | 0 | 15 | 0.757 | 0.298 |
| after | chroma | 1.000 | 1.000 | 0 | 15 | 0.756 | 0.305 |

- **Default case:**
  - before: chroma needed 3 drafts (0.182, 0.185, 0.214), and keyword passed on the first try with 0.212;
  - after: both passed on the first try (0.695 chroma).
- **Prompt alone:** under the old cards-only rule, 14 of 15 first drafts from the new prompt would also pass,
  so the prompt change alone helps.
- **Margin:** after the change, the lowest reviewer score of any LLM draft is 0.633, against the 0.4
  threshold.
- **Caveat:** LLM output varies from run to run, and this is a single 15-case batch per retriever.
- **Older numbers:** the 30-case evaluations elsewhere in this repo (`scripts/run_llm_eval.py`,
  `scripts/compare_retrieval.py`) used the earlier cards-only score with the 0.2 threshold and were not rerun.

## LangSmith tracing (optional)

`ihds/tracing.py` turns tracing on only when a key is present: `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`) in
the environment, or `LANGSMITH_API_KEY` in a `[langsmith]` secrets table. It then:

- sets `LANGSMITH_TRACING` and `LANGCHAIN_TRACING_V2` to `true`, unless one is explicitly `false`;
- sets the project (`LANGSMITH_PROJECT` / `LANGCHAIN_PROJECT`, default `ihds`).

**What a run looks like:** each `agents.run()` is one root run, `ihds-assessment`.

- **Children:** LangGraph, every node, the four tool calls and each writer LLM call. The writer call is an
  `llm` run named `writer_llm` (tags `writer`, `attempt:<n>`) under the `writer_agent` node, with prompt,
  completion and total tokens. LangSmith sums them into the root's `total_tokens`.
- **Tool nesting:** the MCP tool coroutine runs on the client's own event-loop thread, so `MCPToolBackend.call`
  passes the caller's runnable config explicitly. Without that, the tool runs would become separate root
  traces.
- **Tags:** `ihds`, `band:<band>`, `status:<status>`, `draft:<source>`, `retrieval:<method>`.
- **Metadata:** `risk_band`, `final_status`, `draft_source`, `retrieval`, `escalated`, `revisions`,
  `probability`, `tool_transport`, `approved`, `llm_calls`, `llm_total_tokens` and `hostname`. Use
  `hostname` to tell Streamlit Cloud runs from local ones.
- **Run ID and link:** the process that ran the graph sets `AdvisoryResult.langsmith_run_id` and
  `langsmith_url`. That process is the app, or the REST API (embedded or external), which returns both in
  `POST /assess`. The URL comes from one cached project lookup (`tracing.run_url`).
  - The agent trace panel shows **View trace in LangSmith**.
  - Without a URL it shows the run id and "link unavailable".
  - If tracing is on in the app but the run came back without an id, it says so.
- **Tokens in the app:** the draft caption shows `LLM tokens: N`, summed from the responses' usage metadata.
- **Sidebar:** shows `LangSmith tracing: on (project ihds)` or off.
- **Env cache:** `configure()` clears langsmith's cached environment lookups. A check made before
  `configure()` ran would otherwise keep tracing off for the whole process. That happened in the test suite.
- **Zero tokens:** root runs tagged `draft:template` with `total_tokens = 0` come from template mode, when no
  `OPENAI_API_KEY` is set. Those runs make no LLM call. For example, the five newest root runs on 2026-11-25
  before this change were local box runs without a key.

**Keeping the key out of other processes:**

- The MCP stdio child env is an allow-list (PATH, PYTHONPATH, HOME, IHDS_* retrieval settings), so the key
  never reaches the tool server.
- `scripts/start_all.sh` unsets it for the MCP HTTP server.

**Streamlit Cloud secrets:** go to share.streamlit.io, open the app's **⋮ → Settings → Secrets**, add this and
save:

```toml
[langsmith]
LANGSMITH_API_KEY = "lsv2_<your LangSmith API key>"
LANGSMITH_PROJECT = "ihds"
```

**Verified on the box (2026-11-25):** one assessment through MCP stdio tools with gpt-4o-mini, then the
langsmith client API.

- **Root run:** `ihds-assessment` with tags `ihds, band:high, status:ok, draft:llm, retrieval:chroma` and
  metadata `risk_band=high, final_status=ok, draft_source=llm, retrieval=chroma, tool_transport=mcp-stdio`.
- **Runs in the trace:** 21: 16 chain, 1 llm, 4 tool. They were `ihds-assessment`, `LangGraph`, `safety_guard`,
  `route_after_guard`, `supervisor` ×3, `route_from_supervisor` ×3, `risk_agent`, `risk_score`,
  `explain_contributions`, `evidence_agent`, `rag_search`, `population_context`, `writer_agent`, `ChatOpenAI`,
  `reviewer`, `route_after_review` and `finalize`.
- **Stray root runs:** none. No other root runs appeared in the project in that window, so the tools did not
  leak out as separate traces.
- **Link:** `tracing.run_url` returned a link.
- **Live test:** `tests/test_tracing.py::test_live_run_is_traced_with_tags_and_metadata` passed with the key.

**Verified on the box after the writer change (2026-11-25):**

- **Embedded REST API route** (gpt-4o-mini, Chroma):
  - `POST /assess` returned the run id, a `langsmith_url` ending in `/r/<run id>?poll=true`, `llm_tokens=592`
    and grounding 0.691;
  - the trace had 21 runs: 16 chain, 1 llm, 4 tool;
  - the `writer_llm` run had 592 tokens (385 prompt + 207 completion), and the root `ihds-assessment` had
    `total_tokens=592`.
- **Live test:** `test_live_llm_runs_are_children_of_the_root_with_tokens` (needs both keys) passed. It checks
  that every LLM run in the trace is `writer_llm` with tokens, that the root's total covers them, and that the
  metadata `llm_total_tokens` matches.

## Assessment audit log (Postgres / SQLite)

`ihds/audit_log.py` writes one row per completed assessment so runs can be listed later. It uses SQLAlchemy 2
with the psycopg 3 driver. The table is `assessments`:

- `id`, `created_at`
- `risk_probability`, `risk_band`, `reviewer_status`, `draft_source`, `revisions`, `escalated`, `blocked`
- `retrieval_method`, `workflow_route`, `langsmith_run_id` (nullable)
- `features` (JSONB): the eight model inputs only. The clinician question and the draft text are not stored.

**Where it writes**

1. `DATABASE_URL` in the environment, or
2. the Streamlit secret `[database] url`, or
3. if neither is set, a local SQLite file at `data/audit.db` (or `IHDS_AUDIT_SQLITE`; git-ignored).

The table is created on startup. A failed write is logged with credentials redacted, counted on
`ihds_audit_writes_total{result="ok"|"failed"}`, and never raised into the assessment path. The sidebar shows
`Audit log: postgres (neon) · N rows` when the host is Neon, `postgres` for any other Postgres URL, or
`sqlite (local)`. The **Audit log** tab lists the last 20 rows. `GET /assessments/recent?limit=` returns the same
rows as JSON. `POST /assess` writes the row (so the embedded Streamlit API and the Docker/Azure API both log).
The in-process Streamlit fallback also writes, with `workflow_route=in-process`.

The database URL from a Streamlit secret is not copied into `os.environ`, so the MCP stdio child does not inherit
it. `scripts/start_all.sh` unsets `DATABASE_URL` for the MCP HTTP server.

**Local Postgres:** `docker-compose.monitoring.yml` starts `postgres:17` (volume `ihds-pgdata`, healthcheck,
port `127.0.0.1:5432`) and sets the API service's `DATABASE_URL` to `postgresql://ihds:ihds@postgres:5432/ihds`
(local-only credentials, not for production). The API waits until Postgres is healthy.

**Streamlit Community Cloud:** add this in **⋮ → Settings → Secrets** (paste the pooled URL from the Neon
console, project `ihds` → Connect) and reboot the app. Until the secret is set, the Cloud app uses ephemeral
SQLite, which does not survive a reboot.

```toml
[database]
url = "postgresql://USER:PASSWORD@HOST/ihds?sslmode=require"
```

`.env.example` has the same placeholder for a local `DATABASE_URL`. Never commit a real connection string.

**Verified on the box (2026-11-25):** SQLite unit tests in `tests/test_audit_log.py` passed. The Neon project
`ihds` (Postgres 17, aws-us-west-2, database `ihds`, role `ihds_app`) responded through the Neon API
(`SELECT current_database()`). A direct `psycopg` connection from this box to Neon on port 5432 timed out
during the TLS handshake (TCP was accepted by a 198.18.0.0/15 intercept), so no audit row was written from
here. The connection string is not in the repo. Streamlit Cloud uses Neon only after the `[database]` secret
is added.

## GitHub Actions (CI/CD) — how to turn it on

`docs/ci.yml.example` defines the workflow `ci-cd`:

- **lint:** `ruff check .`, configured in `ruff.toml`.
- **test:** `pip install -r requirements-dev.txt` and `pytest -q`.
- **data-validation:** `pip install -r requirements-mlops.txt`, then the GX gate, `pytest tests/test_mlops.py`
  and `dvc repro`, uploading the reports.
- **docker:** `docker build`, then a smoke test of `/health` and `/metrics`.
- **monitoring-stack:** on `ubuntu-latest`:
  - `docker compose -f docker-compose.monitoring.yml up --wait` (unmodified, bridge network);
  - 20 `/assess` calls;
  - asserts `ihds-api` is `up` in `/api/v1/targets` and `sum(ihds_predictions_total) >= 20`;
  - checks Grafana `/api/health`, datasource health and the dashboard.
- **deploy** (optional): runs only through manual `workflow_dispatch` with `deploy=true`, and needs the secrets
  `AZURE_CREDENTIALS`, `ACR_NAME` and `AZURE_RESOURCE_GROUP`. It has never been run. Vocareum lab accounts
  usually cannot create the service principal it needs. The live app was deployed with
  `deploy/azure/deploy_aca.sh`.

**Status:** the workflow is active. It was added through the GitHub web UI as `.github/workflows/ci.yml`
(commit `890ddc9`), with the contents of `docs/ci.yml.example` as of `aaa9649`. Its first run, on 2026-11-25 at
05:52 UTC (Actions run 45556540515), passed lint, test, data-validation and docker. The deploy job was skipped,
as designed.

The active workflow does **not** yet contain the `monitoring-stack` job. The token used to push this repo has no
`workflow` scope, so the job can only be added through the web UI:

1. Open `.github/workflows/ci.yml` on github.com and click the pencil (**Edit**).
2. Replace the contents with the current `docs/ci.yml.example`.
3. Commit to `main`.

Until then, the `monitoring-stack` job has only been rehearsed by hand on the box (see above). It has not run on
GitHub's runners.
