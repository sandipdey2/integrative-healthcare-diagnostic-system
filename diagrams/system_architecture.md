# System Architecture

```
[Patient Features / Imaging Summary]
              |
              v
[Classification Module]  <-- breast-cancer-classification
   Logistic Regression (Wisconsin Diagnostic)
   Probability + feature signals
              |
              v
[Agentic RAG Orchestrator]  <-- agentic-rag-system
   LangGraph + tools + guardrail + reviewer
   Retrieves context, checks consistency
              |
              v
[Generative Evaluation Layer]  <-- generative-mnist concepts
   Synthetic edge cases for stress testing
              |
              v
[Evidently AI Monitoring]
   Drift and performance reports
              |
              v
[Clinician Review]
   Advisory draft + audit trail
```

**Explicit prior-project mapping**
- Classification stage reuses the modeling approach from https://github.com/sandipdey2/breast-cancer-classification
- Orchestration and reviewer logic adapted from https://github.com/sandipdey2/agentic-rag-system
- Synthetic case generation informed by https://github.com/sandipdey2/generative-mnist
