# Verified facts for the P8 deck, defense Q&A and module_summary.pdf

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
| statistical-analysis-project | Welch t-test | ihds/stats_analysis.py | population_context |
| breast-cancer-classification | StandardScaler + LogisticRegression(max_iter=1000, rs=42), metrics | ihds/model.py | risk_score, explain_contributions |
| generative-mnist | MLP GAN, LeakyReLU 0.2, BCE, Adam 2e-4 (0.5, 0.999) → NumPy tabular GAN | ihds/gan.py | synthetic_neighbors |
| agentic-rag-system | LangGraph StateGraph, input guard, TF-IDF RAG, reviewer + grounding, gpt-4o-mini | ihds/agents.py, ihds/workflow.py | rag_search |
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
- Writer: gpt-4o-mini (langchain-openai, temperature 0) only if OPENAI_API_KEY; else deterministic template.
- Typed state (TypedDict) with append-only trace and tool_calls; tools are LangChain StructuredTools (in-process).
- Diagram exported from code: docs/architecture.mmd/.png; test asserts they match.

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

## Tests
- 42 pytest tests, all passing (py3.11 and fresh py3.12 venv); Docker image builds, healthcheck healthy.

## Honest-scope statements (use these words)
- Rule-based supervisor; LLM only writes text and is optional.
- In-process LangChain tools; no MCP server; single process; per-request state only (no long-term memory).
- Offline runs use the template writer, so the revision loop is exercised in tests/evaluation with a stand-in LLM.
- No live gpt-4o-mini call has been verified on the box (no key used); the LLM path is unit-tested with fake models.

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
