# Capstone Defense — Q&A Preparation

**Healthcare AI Diagnostic Support: multi-agent diabetes-risk advisory system (advisory only)**  
Sandip Dey · Repo: `sandipdey2/integrative-healthcare-diagnostic-system` (branch `main`) · Template-mode numbers from `python -m ihds.pipeline` (seed 42); live LLM numbers from `scripts/run_llm_eval.py`

## 60-second elevator pitch

> Type 2 diabetes often goes undetected in primary care, and clinicians have very little time to interpret routine measurements. I built a LangGraph multi-agent system that turns eight routine measurements into an **advisory draft for a clinician**. It provides a risk probability and band, the exact factors driving it, cited guideline snippets, a robustness check for borderline cases and a full audit trace. It integrates four of my earlier projects: Welch t-tests from my Statistical Analysis project, the logistic-regression pipeline from Breast Cancer Classification, a GAN from Generative MNIST and the guarded LangGraph RAG pattern from Agentic RAG. A safety guard blocks unsafe requests, a reviewer checks every draft and can send it back at most twice, and Evidently AI monitors both the input data and the agents' outputs. On 154 held-out patients, the system reaches ROC-AUC 0.813 and recall 0.70, every released draft is cited and carries a disclaimer, and Evidently catches a simulated glucose-sensor drift. The system never diagnoses. It is a transparent, tested assistant that leaves the decision with the clinician.

## A. Problem and scope

**1. What problem are you solving, and for whom?**
The system supports diabetes-risk screening and triage in primary care. Users are clinicians, not patients. The output is an advisory draft with a probability, a risk band (low < 0.30, moderate, high ≥ 0.60), the drivers, cited evidence and a trace. It does not provide a diagnosis.

**2. Why is AI appropriate here?**
The task combines numeric risk estimation (ML), evidence retrieval and explanation (RAG and text generation) and safety checks. A single rule or model cannot cover all of these. Screening is also a setting where the cost of a recommendation is low because it leads to a confirmatory lab test, so an advisory tool is appropriate.

**3. What data did you use, and what are its limits?**
I used the Pima Indians Diabetes Database (Smith et al., 1988) from OpenML #37: 768 women aged 21 and over of Pima heritage, 268 (34.9%) diabetic, with eight features. Impossible zeros are treated as missing: Insulin 48.7%, SkinThickness 29.6%, BloodPressure 4.6%, BMI 1.4% and Glucose 0.7%. The main limitations are one population, one sex and a small sample.

## B. Architecture and design

**4. Walk me through the architecture.**
`START → safety_guard → (refuse | supervisor)`. The rule-based supervisor sends the case to `risk_agent`, then `evidence_agent`, then `uncertainty_agent` (only when 0.40 ≤ p < 0.60), and each one returns to the supervisor. It then calls `writer_agent → reviewer`. The reviewer either approves the draft and sends it to `finalize`, or sends it back to the writer at most twice. Shared state is a typed `AgentState` with append-only `trace` and `tool_calls`. The diagram `docs/architecture.png` is exported from the compiled graph, and a test checks that it matches.

**5. Why LangGraph?**
It provides explicit nodes, conditional edges, typed shared state with reducers and a recursion limit (30). That makes the control flow inspectable, testable and exportable as a diagram. I had already used it in my Agentic RAG project, so I reused a pattern I had validated.

**6. Why is the supervisor rule-based instead of LLM-driven?**
The routing logic is simple and safety-relevant: risk, then evidence, then uncertainty only when needed, then the writer. Rules make every path deterministic and unit-testable (20 offline agent tests), cost nothing and work offline. LLM routing would add latency, cost and non-determinism without a clear benefit here. The tradeoff is less flexibility for unanticipated requests.

**7. Is this really multi-agent if the supervisor is rules?**
Yes. In the sense LangGraph uses, it has separate specialist nodes with their own tools and responsibilities, a router and a critic that can reject work. I am careful not to overclaim: the agents are nodes in one process sharing one state. They are not autonomous services, and the LLM only writes text.

**8. Why cap the reviewer loop, and why at 2?**
Without a cap, a misbehaving LLM could loop forever or until the recursion limit. `MAX_REVISIONS = 2` means the writer runs at most three times. After that, `finalize` releases the deterministic template, which always passes review, and sets `escalated=True` for human review. I tested this with an always-unsafe stand-in LLM: three attempts, three rejections, template released and case escalated.

**9. Are the tools served over MCP?**
No. They are LangChain `StructuredTool`s called in-process: `risk_score`, `explain_contributions`, `rag_search`, `population_context` and `synthetic_neighbors`. There is no MCP server, gateway or message bus. MCP would only make sense if the tools had to be shared across separate services.

**10. Does the system have memory?**
It keeps only per-request state. The trace and tool log are accumulated within one run and shown in the UI. There is no long-term or cross-session memory, which is intentional because I do not want to store patient inputs.

**11. Where is the LLM used?**
Only in `writer_agent`. It uses gpt-4o-mini through `langchain-openai` (temperature 0) when `OPENAI_API_KEY` is set; otherwise it uses a deterministic template. It honors `OPENAI_API_BASE`, so it runs through Vocareum's OpenAI-compatible endpoint. I verified it live: four Vocareum tests pass (ping, full graph run with the LLM writer approved, unsafe request refused before any LLM call), and a 30-case live evaluation is summarised in T2. Scenario numbers elsewhere come from reproducible template runs.

## C. Integration of prior projects

**12. Which prior projects did you integrate, and how?**

- **statistical-analysis-project**: Welch t-tests in `ihds/stats_analysis.py`, exposed through the `population_context` tool.
- **breast-cancer-classification**: StandardScaler + LogisticRegression (max_iter=1000, random_state=42) and the same metric set in `ihds/model.py`, exposed through `risk_score` and `explain_contributions`.
- **generative-mnist**: the MLP GAN design (LeakyReLU 0.2, BCE, Adam 2e-4) re-implemented in NumPy for tabular data in `ihds/gan.py`, exposed through `synthetic_neighbors`.
- **agentic-rag-system**: the LangGraph StateGraph, input guard, TF-IDF RAG, reviewer with grounding score and trace logging in `ihds/agents.py` and `ihds/workflow.py`.

**13. Isn't this just four projects stitched together?**
No. Each prior method is a tool with a defined clinical role inside one graph, and they depend on each other. The risk output decides whether the GAN-based uncertainty check runs, the evidence agent queries population statistics for the top risk driver, and the reviewer checks that the writer cites what the evidence agent retrieved. New work includes the dataset, the supervisor/specialist design, the uncertainty agent, the capped revision loop with escalation and Evidently monitoring of agent outputs.

**14. Why re-implement the GAN in NumPy instead of PyTorch?**
It keeps the Docker image and Streamlit deployment small. The backpropagation is checked against finite differences in `tests/test_gan.py`. The cost is weaker fidelity on some features.

## D. Model and metrics

**15. Why logistic regression?**
Its per-feature log-odds contributions are exact, so the explanation the clinician sees reflects how the model actually decided. A test checks that the contributions sum to the logit. It performed well in my Breast Cancer project and is easy to audit. Test ROC-AUC is 0.813, and train ROC-AUC is 0.851, so there is little overfitting.

**16. What are the held-out results?**
On n = 154, accuracy was 0.734, precision 0.603, recall 0.704, F1 0.650 and ROC-AUC 0.813. The confusion matrix was TN 75, FP 25, FN 16 and TP 38. The strongest drivers were glucose (OR 3.27 per SD) and BMI (2.03).

**17. Why prioritise recall over precision?**
In screening, a missed diabetic patient (false negative) can lead to years of untreated disease. A false positive usually leads to an inexpensive confirmatory test. `class_weight="balanced"` moves the model toward recall (0.704) at the expense of precision (0.603). With 25 false positives against 16 false negatives, I consider that an acceptable trade for an advisory screening tool.

**18. Why not a stronger model like XGBoost?**
It might gain a few points of AUC on 768 rows, but it would need post-hoc explanations such as SHAP instead of exact contributions, and it carries more overfitting risk. For a clinician-facing advisory tool, I valued faithful explanations more. I would benchmark it as future work.

**19. Are the risk bands meaningful?**
Yes. Observed diabetes rates rise across the bands: low 9.5% (6/63), moderate 47.2% (17/36) and high 59.6% (31/52).

## E. Monitoring and evaluation

**20. What does Evidently monitor?**
It produces six reports: hold-out data drift and classification metrics, a simulated glucose-shifted set, GAN fidelity, text evaluations of the agent drafts, drift in agent outputs and guard classification. They are saved as HTML and JSON in `monitoring/reports/` and shown in the app's Monitoring tab.

**21. What did the drift tests show?**
On hold-out data, 0 of 8 features drifted. In the simulated +25 mg/dL glucose calibration error, Glucose drifted (p < 0.001), precision fell from 0.603 to 0.527 and recall rose to 0.907. The agent-output report also caught the shift: only `probability` drifted (p = 0.0009, mean 0.452 → 0.606), while grounding, revisions, tool calls and length remained stable.

**22. How did you evaluate the agents themselves?**
I ran all 154 test records through the graph: 151 were released and 3 were blocked for missing BMI. The reviewer approved all 151 drafts, with mean grounding 0.551. Evidently's text evaluations showed 100% citation, 100% disclaimer and 0% forbidden phrases. All 27 borderline cases went to the uncertainty agent, and all 500 GAN cases ran without failure.

**23. How were the GAN outputs used, and what are their limits?**
They are used only for stress tests and the uncertainty neighbourhood, never for training. Evidently found that 5 of 8 synthetic features drifted from real data (Pregnancies, SkinThickness, Insulin, DPF and Age), because the GAN does not reproduce the median-imputation spikes. Glucose, BloodPressure and BMI did not drift.

**24. How did you test the system?**
There are 47 pytest tests: 43 offline tests (passing on Python 3.11, 3.12 and 3.14) and 4 live `vocareum` tests that call gpt-4o-mini and are skipped without a key; all 47 pass with the key. They cover data and statistics, the model, GAN gradients, the guard, retrieval and reviewer, graph structure against the documented diagram, routing, every agent node, the loop cap, LLM fallback, Evidently reports, the end-to-end pipeline and the Streamlit app through `AppTest`. The Docker image builds and passes its healthcheck. The details are in `TESTING.md`.

## F. Safety, ethics and deployment

**25. How does the safety guard work, and how good is it?**
It is rule-based. It checks ranges and required fields, and uses regex patterns to catch prompt injection and dosing or diagnosis requests. On 19 labelled cases it scored accuracy 0.947, precision 1.0 and recall 0.889. It missed one case, "Should she stop taking my medication?", because of paraphrase brittleness.

**26. What are the main ethical risks?**

- **Bias:** the model was trained only on Pima women and is not validated for other groups.
- **Automation bias:** a clinician may over-trust a number, as in the p = 0.077 false negative.
- **Accountability:** the clinician decides, the output is advisory and escalations are flagged.
- **Transparency:** exact contributions, cited evidence and a full trace are shown.
- **Misuse:** the guard and reviewer block diagnosis and dosing language.

**27. What would be required for real clinical deployment?**
It would need external and subgroup validation, an intent classifier instead of keyword rules, live drift alerts with owners, clinical governance of the knowledge base, authentication and audit logging, privacy compliance and a regulatory assessment under FDA clinical-decision-support guidance.

**28. How is it deployed?**
The Streamlit app (`app.py`) can be deployed to Streamlit Community Cloud from `main` (Python 3.14 on Cloud; pins install from wheels on 3.11–3.14), with optional `OPENAI_API_KEY` and `OPENAI_API_BASE` secrets. For Azure, a `python:3.11-slim` Docker image on port 8501 is deployed with `deploy/azure/deploy_aca.sh` (`az acr build` + `az containerapp create`). With `WITH_OPENAI=1` the script stores the key as a Container Apps secret and sets `OPENAI_API_BASE` as an environment variable. An ACI alternative and a cleanup script are included. Evidently reports are generated during the image build.

## G. Failure cases, limitations and next steps

**29. Describe a failure case.**
The confident false negative: a diabetic patient with glucose 78, BMI 31 and age 26 received p = 0.077. Low glucose dominates the linear model. This is why the output is advisory and shows the drivers, so a clinician can see that the low risk depends almost entirely on one value.

**30. What are the main limitations?**
There is one small dataset from one population, the hand-written knowledge base has only 14 rows, the keyword guard can be bypassed by paraphrase, the GAN has fidelity gaps, live LLM drafts are less grounded than the template (mean 0.303 vs 0.549) and the routing is rule-based. Also, there is no EHR integration, authentication or persistent storage.

**31. What would you do next?**

1. Validate on external, multi-ethnic data and report subgroup metrics.
2. Replace the keyword guard with an embedding or LLM intent classifier and add the paraphrase cases.
3. Add claim-level citation checks to the reviewer, since one live draft cited a real snippet for an unrelated claim.
4. Schedule Evidently drift checks on production inputs, with alerting.
5. Calibrate probabilities and tune the threshold together with clinicians.
6. Expand the knowledge base with clinical review.

## Tricky questions

**T1. "Your diagram looks like an agent framework, but isn't it just a pipeline with if-statements?"**
The routing is rule-based, and I say so. What makes it agentic is the structure: separate tool-using specialists, a shared typed state, a hub-and-spoke supervisor that decides from state, and a critic that can reject work and trigger a bounded loop. The diagram is exported from the compiled graph and tested, so it shows exactly what runs.

**T2. "Did you ever run the LLM?"**
Yes, live, through Vocareum's endpoint. On 30 held-out cases, gpt-4o-mini drafts were approved first time in 23 cases and after one revision in 5; 2 hit the revision cap and were escalated with the template. All 11 rejections were for low grounding. Evidently showed 100% citation and disclaimer rates and no forbidden phrases, but mean grounding of 0.303 against 0.549 for the template, and 4.7 s against 0.01 s per case. So the reviewer loop did real work: it caught weakly grounded drafts the model produced on its own.

**T3. "A precision of 0.60 means 40% of positive flags are wrong. Is that acceptable?"**
For an advisory screening prompt that leads to a cheap confirmatory test, yes. That is the recall-first trade-off. For a diagnostic claim, no, and the system makes no diagnostic claim.

**T4. "Why use a GAN at all if 5 of 8 features drift?"**
Because I measured it and limited its role accordingly. It gives a plausible neighbourhood for the main drivers (glucose, BMI and blood pressure do not drift) for robustness checks and a 500-case stress test. It never trains the model.

**T5. "Your guard missed a medication question. Isn't that a safety failure?"**
It is a real gap, and I report it. In this case, there was layered protection: the writer never produces medication advice, and the reviewer rejects banned phrases. Still, the request should have been refused, and I plan to fix it with a learned intent classifier.

**T6. "Wouldn't an LLM supervisor be more 'agentic'?"**
More autonomous, yes, but less predictable and harder to audit in a clinical workflow. I chose determinism where safety matters and limited the LLM to the part where language fluency helps.

**T7. "How do you know the diagram is accurate?"**
`tests/test_agents.py::test_graph_structure_matches_docs` compares `docs/architecture.mmd` with `graph.get_graph().draw_mermaid()` and fails if they differ.
