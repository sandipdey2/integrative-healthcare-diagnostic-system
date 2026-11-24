import streamlit as st

st.set_page_config(
    page_title="Healthcare AI Diagnostic Support",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare AI Diagnostic Support System")
st.caption("Project 7 — Integrative Industry Synthesis | Advisory only — clinician review required")

st.markdown("""
This system integrates methods from three prior capstone projects:

- **Classification** → [breast-cancer-classification](https://github.com/sandipdey2/breast-cancer-classification)  
  Logistic regression on the Breast Cancer Wisconsin (Diagnostic) dataset
- **Agentic RAG** → [agentic-rag-system](https://github.com/sandipdey2/agentic-rag-system)  
  LangGraph orchestration, tool calling, guardrails, and reviewer checks
- **Generative techniques** → [generative-mnist](https://github.com/sandipdey2/generative-mnist)  
  Synthetic edge-case generation concepts for evaluation
""")

tab1, tab2, tab3, tab4 = st.tabs(["Advisory Analysis", "Architecture", "Prior Projects", "About"])

with tab1:
    st.header("Generate Advisory Draft")
    col1, col2 = st.columns([2, 1])

    with col1:
        case_id = st.text_input("Case / Image ID", value="demo-001")
        notes = st.text_area("Optional clinical notes", height=100)

    with col2:
        st.metric("System Status", "Ready")
        st.metric("Human Review", "Required")

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running classification + agentic retrieval..."):
            # Placeholder results — replace with real model/agent calls
            st.success("Analysis complete")
            st.subheader("Advisory Findings")
            st.write("**Primary finding:** Possible consolidation in the right lower lobe")
            st.write("**Classification probability (malignant/elevated):** 0.82")
            st.write("**Supporting context retrieved:** 2 relevant prior notes")
            st.write("**Reviewer check:** Consistent with retrieved evidence")

            st.warning("""
            **Important:** This is an advisory draft only.  
            It does not constitute a diagnosis or clinical decision.  
            Clinician review and sign-off are required before any action.
            """)

            with st.expander("Confidence & Limitations"):
                st.write("- Model adapted from breast-cancer-classification pipeline")
                st.write("- Agentic layer based on agentic-rag-system design")
                st.write("- Synthetic evaluation cases informed by generative-mnist concepts")
                st.write("- Output intended for clinician review only")

with tab2:
    st.header("System Architecture")
    st.markdown("""
    ```
    [Patient Features / Case Input]
              |
              v
    [Classification Module]          ← breast-cancer-classification
       Logistic Regression
       Probability + feature signals
              |
              v
    [Agentic RAG Orchestrator]       ← agentic-rag-system
       LangGraph + tools + reviewer
       Context retrieval + consistency check
              |
              v
    [Generative Evaluation Layer]    ← generative-mnist concepts
       Synthetic edge cases
              |
              v
    [Monitoring]
       Evidently AI drift reports
              |
              v
    [Clinician Review]
       Advisory draft + audit trail
    ```
    """)

with tab3:
    st.header("Integrated Prior Projects")
    st.markdown("""
    | Project | Repository | Role in this system |
    |---------|------------|---------------------|
    | breast-cancer-classification | [link](https://github.com/sandipdey2/breast-cancer-classification) | Core classification & scoring |
    | agentic-rag-system | [link](https://github.com/sandipdey2/agentic-rag-system) | Orchestration, retrieval, reviewer |
    | generative-mnist | [link](https://github.com/sandipdey2/generative-mnist) | Synthetic evaluation concepts |
    """)

with tab4:
    st.header("About This Artifact")
    st.markdown("""
    **Project 7 — Integrative Industry Synthesis**  
    Healthcare AI diagnostic support system designed for advisory use only.

    - Human-in-the-loop is mandatory  
    - All outputs include confidence and limitations  
    - Designed for auditability and responsible deployment  

    [View full repository](https://github.com/sandipdey2/integrative-healthcare-diagnostic-system)
    """)

st.sidebar.title("Navigation")
st.sidebar.markdown("**Advisory system only**")
st.sidebar.markdown("Clinician review required")
st.sidebar.divider()
st.sidebar.markdown("[GitHub Repo](https://github.com/sandipdey2/integrative-healthcare-diagnostic-system)")
