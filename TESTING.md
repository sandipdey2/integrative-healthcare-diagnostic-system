# Testing — final test plan and results

Run date: 2026-11-24. Environment: Python 3.11.x (also re-run in a fresh Python 3.12 venv from
`requirements-dev.txt`: same result). No API key set (offline mode).

```bash
pip install -r requirements-dev.txt
pytest -q              # -> 42 passed
python -m ihds.pipeline
```

## 1. Automated tests (pytest) — 42 / 42 passed

| File | Tests | Covers | Result |
|---|---|---|---|
| `tests/test_data_stats.py` | 4 | dataset shape (768×9, 268 positive), impossible-zero cleaning, stratified split (614/154), Welch t-test ranking | 4 passed |
| `tests/test_model.py` | 4 | test ROC-AUC > 0.78 and recall > 0.6, monotonic in glucose, contributions sum exactly to the logit, imputation of missing optional values | 4 passed |
| `tests/test_gan.py` | 3 | hand-written backprop vs finite differences, valid/finite/non-collapsed samples, glucose mean within 10 mg/dL after training | 3 passed |
| `tests/test_workflow.py` | 9 | guard (ranges, required fields, injection, dosing, documented paraphrase gap), zero normalisation, TF-IDF retrieval, grounding score, template passes review, reviewer rejects unsafe text | 9 passed |
| `tests/test_agents.py` | 19 | **graph structure** equals the exported `docs/architecture.mmd`; **routing** functions and supervisor order; non-borderline, borderline (uncertainty agent) and blocked paths; **revision loop cap** (3 writer attempts → escalation); loop recovery after one rejection; LLM error fallback; gpt-4o-mini selection; **each agent node** (safety_guard, risk, evidence, uncertainty, writer, reviewer, finalize) | 19 passed |
| `tests/test_monitoring.py` | 1 | Evidently holdout/shifted/synthetic reports written (HTML+JSON); no drift on holdout; Glucose drift and lower precision on the shifted set | 1 passed |
| `tests/test_e2e.py` | 2 | full pipeline end-to-end incl. Evidently agent evaluation; Streamlit app runs and the "Run advisory workflow" button works with no API key (`streamlit.testing.v1.AppTest`) | 2 passed |

The LLM writer path is tested with LangChain's `FakeListChatModel` and a broken client (no live OpenAI call in tests).

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
| Headless Streamlit | serves on 8501 without a key | HTTP 200, health `ok` | ✅ |

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

- [ ] Streamlit Cloud URL opens and "Run advisory workflow" returns a draft.
- [ ] Azure Container App URL opens; `curl https://<fqdn>/_stcore/health` returns `ok`.
- [ ] Optional: with `OPENAI_API_KEY` set, the draft source shows `llm` and the reviewer approves it.
