# Project 7 — Integrative Industry Synthesis
## Healthcare AI Diagnostic Support System

### Prior Capstone Projects Integrated (minimum 3 required)
This artifact explicitly integrates methods and insights from the following completed projects:

1. **breast-cancer-classification**  
   https://github.com/sandipdey2/breast-cancer-classification  
   Supervised logistic regression on the Breast Cancer Wisconsin (Diagnostic) dataset. Provides the core classification and probability scoring component.

2. **agentic-rag-system**  
   https://github.com/sandipdey2/agentic-rag-system  
   LangGraph multi-agent orchestration, tool calling, guardrails, reviewer checks, and RAGAS-style evaluation. Provides the explanation and retrieval layer.

3. **generative-mnist**  
   https://github.com/sandipdey2/generative-mnist  
   GAN-based synthetic sample generation on MNIST. Informs synthetic edge-case generation and evaluation practices.

Additional practices from **statistical-analysis-project** inform exploratory analysis of score distributions.

### Submission Contents
- `docs/module_summary.pdf` — Reflective Synthesis Paper (1,500+ words, required sections, APA 7, scholarly source)
- `notebooks/integrated_diagnostic_system.ipynb` — New integrative notebook
- `diagrams/system_architecture.md` — Architecture with component mapping
- `requirements.txt`
- `README.md`

### Integration Flow
Classification scores (breast-cancer-classification) → Agentic RAG workflow (agentic-rag-system) → Synthetic evaluation cases (generative-mnist concepts) → Evidently AI monitoring → Clinician review

### System Boundaries
- Advisory outputs only
- Human-in-the-loop required
- All results include confidence and limitations
