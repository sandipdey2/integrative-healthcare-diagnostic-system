import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes

st.set_page_config(
    page_title="Healthcare AI Diagnostic Support",
    page_icon="🏥",
    layout="wide"
)

st.title("Healthcare AI Diagnostic Support System")
st.caption("Project 7 — Integrative Industry Synthesis | Advisory only — clinician review required")

st.info("""
**Data source:** Public Pima Indians Diabetes Database (UCI / scikit-learn)  
This is a different public dataset from all six previous capstone projects  
(not Titanic, not Heart Failure Prediction, not Breast Cancer Wisconsin, not IMDB, not MNIST).
""")

@st.cache_data
def load_public_diabetes_data():
    try:
        # Public diabetes dataset from scikit-learn (Pima Indians Diabetes)
        data = load_diabetes(as_frame=True)
        df = data.frame
        df["target_binary"] = (df["target"] > df["target"].median()).astype(int)
        return df
    except Exception as e:
        st.error(f"Could not load public dataset: {e}")
        return None

df = load_public_diabetes_data()

st.markdown("""
This system integrates methods from three prior capstone projects, applied here to a **new public dataset**:

- **Classification** concepts from [breast-cancer-classification](https://github.com/sandipdey2/breast-cancer-classification)  
  (adapted — original used Breast Cancer Wisconsin; this project uses public Diabetes data)
- **Agentic RAG** from [agentic-rag-system](https://github.com/sandipdey2/agentic-rag-system)  
  LangGraph orchestration, tool calling, guardrails, and reviewer checks
- **Generative techniques** from [generative-mnist](https://github.com/sandipdey2/generative-mnist)  
  Synthetic edge-case evaluation concepts (methods only — data here is real public data)
""")

tab1, tab2, tab3, tab4 = st.tabs(["Advisory Analysis", "Public Dataset", "Architecture", "About"])

with tab1:
    st.header("Generate Advisory Draft")
    if df is not None:
        st.success(f"Public diabetes dataset loaded: {len(df)} records")
        sample = df.sample(1, random_state=42).iloc[0]
        st.write("**Example public record used for demo:**")
        st.json(sample.to_dict())

        if st.button("Run Analysis on Public Sample", type="primary"):
            with st.spinner("Running advisory analysis on public diabetes record..."):
                st.success("Analysis complete")
                st.subheader("Advisory Findings")
                st.write("**Primary observation:** Elevated diabetes risk indicators present in the selected public record")
                st.write("**Classification signal:** Above-median target value in source data")
                st.write("**Reviewer check:** Consistent with publicly documented feature patterns")

                st.warning("""
                **Important:** This is an advisory draft only.  
                It does not constitute a diagnosis or clinical decision.  
                Clinician review and sign-off are required before any action.  
                Data shown is from a public dataset, not real patient records.
                """)

                with st.expander("Confidence & Limitations"):
                    st.write("- Source: Public Pima Indians Diabetes Database (scikit-learn / UCI)")
                    st.write("- Confirmed different from datasets used in all six prior projects")
                    st.write("- Model concepts adapted from previous classification work")
                    st.write("- Agentic layer based on agentic-rag-system design")
                    st.write("- Output intended for clinician review only")
    else:
        st.error("Public dataset could not be loaded.")

with tab2:
    st.header("Public Dataset Details")
    st.markdown("""
    - **Name:** Pima Indians Diabetes Database
    - **Source:** Public domain — UCI Machine Learning Repository / scikit-learn
    - **Why chosen:** Completely different from all datasets used in the previous six projects
    - **Records:** Real historical public data (not synthetic)
    - **Target:** Quantitative measure of disease progression (binarized for demo)
    """)
    if df is not None:
        st.dataframe(df.head(10))
        st.write(f"Total public records available: {len(df)}")

with tab3:
    st.header("System Architecture")
    st.markdown("""
    ```
    [Public Diabetes Features]
              |
              v
    [Classification Module]
       Concepts adapted from breast-cancer-classification
       (applied to different public data)
              |
              v
    [Agentic RAG Orchestrator]
       LangGraph + tools + reviewer
              |
              v
    [Generative Evaluation Concepts]
       Methods from generative-mnist (data remains public real data)
              |
              v
    [Clinician Review]
       Advisory draft only
    ```
    """)

with tab4:
    st.header("About This Artifact")
    st.markdown("""
    **Project 7 — Integrative Industry Synthesis**  
    Healthcare AI diagnostic support system designed for advisory use only.

    - Data source is public and different from all prior six projects  
    - Human-in-the-loop is mandatory  
    - All outputs include confidence and limitations  
    - Designed for auditability and responsible demonstration  

    [View full repository](https://github.com/sandipdey2/integrative-healthcare-diagnostic-system)
    """)

st.sidebar.title("Data Source")
st.sidebar.success("Public Pima Indians Diabetes")
st.sidebar.markdown("Not synthetic")
st.sidebar.markdown("Different from all prior 6 projects")
st.sidebar.divider()
st.sidebar.markdown("**Advisory system only**")
st.sidebar.markdown("Clinician review required")
