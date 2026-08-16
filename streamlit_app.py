import json
import os
import requests
import streamlit as st
import pandas as pd


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Enterprise Agentic RAG - LLMOps",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# Configuration
# ============================================================

EVALUATION_FILE = "data/evaluation_results.json"

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data
def load_evaluation_results():

    if not os.path.exists(EVALUATION_FILE):
        return None

    try:
        with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as exc:
        st.error(f"Unable to load evaluation results: {exc}")
        return None


def check_api_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if response.status_code == 200:
            return True, response.json()

        return False, response.text

    except Exception as exc:
        return False, str(exc)


def ask_api(question):

    try:

        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question
            },
            timeout=120
        )

        if response.status_code == 200:
            return True, response.json()

        return False, response.text

    except Exception as exc:
        return False, str(exc)


# ============================================================
# Header
# ============================================================

st.title("🤖 Enterprise Agentic RAG")
st.subheader("LLMOps Evaluation & Production Monitoring Dashboard")

st.markdown(
    """
    This dashboard provides visibility into the deployed
    Enterprise Agentic RAG system, including evaluation metrics,
    API health, routing behavior and request performance.
    """
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Configuration")

st.sidebar.write(
    f"**API URL:** `{API_URL}`"
)

if st.sidebar.button("🔄 Refresh Dashboard"):

    st.cache_data.clear()
    st.rerun()


# ============================================================
# API Health
# ============================================================

st.header("🟢 Production API Health")

healthy, health_response = check_api_health()

if healthy:

    st.success("API is healthy")

    st.json(health_response)

else:

    st.error("API is unavailable")

    st.code(str(health_response))


# ============================================================
# Evaluation Results
# ============================================================

st.header("📊 RAGAS Evaluation")

evaluation = load_evaluation_results()

if evaluation is None:

    st.warning(
        f"Evaluation file not found: `{EVALUATION_FILE}`"
    )

else:

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = evaluation.get("summary", {})

    total_questions = summary.get(
        "total_questions",
        0
    )

    ragas_evaluated = summary.get(
        "ragas_evaluated",
        0
    )

    ragas_skipped = summary.get(
        "ragas_skipped",
        0
    )

    ragas_errors = summary.get(
        "ragas_errors",
        0
    )


    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Questions",
        total_questions
    )

    col2.metric(
        "RAGAS Evaluated",
        ragas_evaluated
    )

    col3.metric(
        "Skipped",
        ragas_skipped
    )

    col4.metric(
        "Errors",
        ragas_errors
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = evaluation.get("metrics", {})

    st.subheader("Evaluation Metrics")

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    metric_values = []

    for metric in metric_names:

        value = metrics.get(metric)

        if value is not None:

            metric_values.append({
                "Metric": metric.replace("_", " ").title(),
                "Score": round(float(value), 4)
            })


    if metric_values:

        metrics_df = pd.DataFrame(
            metric_values
        )

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            metrics_df.set_index("Metric")
        )

    else:

        st.warning(
            "No evaluation metrics found."
        )


    # --------------------------------------------------------
    # Detailed Evaluation Results
    # --------------------------------------------------------

    st.subheader("Question-Level Evaluation")

    results = evaluation.get(
        "results",
        []
    )

    if results:

        rows = []

        for item in results:

            rows.append({

                "Question": item.get(
                    "question",
                    ""
                ),

                "Route": item.get(
                    "route",
                    ""
                ),

                "Faithfulness": item.get(
                    "faithfulness"
                ),

                "Answer Relevancy": item.get(
                    "answer_relevancy"
                ),

                "Context Precision": item.get(
                    "context_precision"
                ),

                "Context Recall": item.get(
                    "context_recall"
                ),

            })


        results_df = pd.DataFrame(rows)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No question-level results available."
        )


# ============================================================
# API Testing
# ============================================================

st.header("🧪 Production API Test")

question = st.text_input(
    "Enter a question",
    placeholder="What is the main contribution of the Transformer?"
)


if st.button("Ask Enterprise RAG"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Sending request to production API..."
        ):

            success, response = ask_api(
                question
            )


        if success:

            st.success(
                "Request completed successfully"
            )

            # -----------------------------------------------
            # Request information
            # -----------------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Route",
                response.get(
                    "route",
                    "unknown"
                )
            )

            col2.metric(
                "Retrieved Documents",
                response.get(
                    "retrieved_documents",
                    0
                )
            )

            col3.metric(
                "Latency",
                f"{response.get('latency_seconds', 0)} s"
            )


            # -----------------------------------------------
            # Answer
            # -----------------------------------------------

            st.subheader("Answer")

            st.write(
                response.get(
                    "answer",
                    ""
                )
            )


            # -----------------------------------------------
            # Raw response
            # -----------------------------------------------

            with st.expander(
                "View API Response"
            ):

                st.json(response)

        else:

            st.error(
                "API request failed"
            )

            st.code(
                str(response)
            )


# ============================================================
# LLMOps Architecture
# ============================================================

st.header("🏗️ LLMOps Pipeline")

st.markdown(
    """
    **User**
    ↓

    **Streamlit Dashboard**
    ↓

    **FastAPI**
    ↓

    **LangGraph Agentic RAG**
    ↓

    **Vector / Hybrid Retrieval**
    ↓

    **Groq LLM**
    ↓

    **LangSmith Tracing**
    ↓

    **RAGAS Evaluation**
    ↓

    **Evaluation Dashboard**
    """
)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Enterprise Agentic RAG + LLMOps | "
    "FastAPI • LangGraph • LangSmith • RAGAS • Docker • GitHub Actions"
)