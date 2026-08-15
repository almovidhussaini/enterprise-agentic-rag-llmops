import json
from pathlib import Path

import pandas as pd
import streamlit as st

from observability.langsmith_monitor import (
    get_observability_data,
    calculate_summary,
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FILE = Path(
    "data/evaluation_results.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise RAG - LLMOps Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Enterprise Agentic RAG — LLMOps Dashboard"
)

st.caption(
    "RAGAS Evaluation + LangSmith LLM Observability"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Dashboard Settings"
)

observation_hours = st.sidebar.selectbox(
    "Observability Time Window",
    options=[
        1,
        6,
        12,
        24,
        48,
        72,
        168,
    ],
    index=3,
    format_func=lambda x: f"Last {x} hours",
)

observation_limit = st.sidebar.selectbox(
    "Maximum LangSmith Runs",
    options=[
        20,
        50,
        100,
        250,
        500,
    ],
    index=2,
)

st.sidebar.divider()

refresh_dashboard = st.sidebar.button(
    "🔄 Refresh Dashboard"
)

if refresh_dashboard:

    st.cache_data.clear()

    st.rerun()


# ============================================================
# LOAD RAGAS RESULTS
# ============================================================

@st.cache_data
def load_results():

    if not RESULTS_FILE.exists():

        return None

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


data = load_results()


# ============================================================
# CHECK EVALUATION FILE
# ============================================================

if data is None:

    st.error(
        f"Evaluation results not found:\n\n"
        f"{RESULTS_FILE}"
    )

    st.info(
        "Run the evaluation first:"
    )

    st.code(
        "python -m app.evaluation.langsmith_eval"
    )

    st.stop()


# ============================================================
# BASIC INFORMATION
# ============================================================

experiment = data.get(
    "experiment",
    "Unknown"
)

dataset = data.get(
    "dataset",
    "Unknown"
)

timestamp = data.get(
    "timestamp",
    "Unknown"
)

metrics = data.get(
    "metrics",
    {}
)

summary = data.get(
    "summary",
    {}
)

questions = data.get(
    "questions",
    []
)


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

st.subheader(
    "Experiment Information"
)

info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.metric(
        "Experiment",
        experiment,
    )


with info_col2:

    st.metric(
        "Dataset",
        dataset,
    )


with info_col3:

    st.metric(
        "Timestamp",
        timestamp,
    )


st.divider()


# ============================================================
# EVALUATION SUMMARY
# ============================================================

st.subheader(
    "Evaluation Summary"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Questions",
        summary.get(
            "total_questions",
            len(questions),
        ),
    )


with col2:

    st.metric(
        "RAGAS Evaluated",
        summary.get(
            "ragas_evaluated",
            0,
        ),
    )


with col3:

    st.metric(
        "RAGAS Skipped",
        summary.get(
            "ragas_skipped",
            0,
        ),
    )


with col4:

    st.metric(
        "RAGAS Errors",
        summary.get(
            "ragas_errors",
            0,
        ),
    )


st.divider()


# ============================================================
# RAGAS METRICS
# ============================================================

st.subheader(
    "Overall RAGAS Metrics"
)


metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)


def display_metric(
    column,
    name,
    value,
):

    with column:

        if value is None:

            st.metric(
                name,
                "N/A",
            )

        else:

            st.metric(
                name,
                f"{value:.4f}",
            )


display_metric(
    metric_col1,
    "Faithfulness",
    metrics.get(
        "faithfulness"
    ),
)

display_metric(
    metric_col2,
    "Answer Relevancy",
    metrics.get(
        "answer_relevancy"
    ),
)

display_metric(
    metric_col3,
    "Context Precision",
    metrics.get(
        "context_precision"
    ),
)

display_metric(
    metric_col4,
    "Context Recall",
    metrics.get(
        "context_recall"
    ),
)


st.divider()


# ============================================================
# RAGAS METRIC CHART
# ============================================================

st.subheader(
    "RAGAS Metric Comparison"
)


metric_data = {

    "Faithfulness":
        metrics.get(
            "faithfulness"
        ),

    "Answer Relevancy":
        metrics.get(
            "answer_relevancy"
        ),

    "Context Precision":
        metrics.get(
            "context_precision"
        ),

    "Context Recall":
        metrics.get(
            "context_recall"
        ),

}


metric_data = {

    key: value

    for key, value
    in metric_data.items()

    if value is not None

}


if metric_data:

    chart_df = pd.DataFrame(
        {
            "Metric":
                list(
                    metric_data.keys()
                ),

            "Score":
                list(
                    metric_data.values()
                ),
        }
    )

    st.bar_chart(
        chart_df.set_index(
            "Metric"
        )
    )

else:

    st.info(
        "No RAGAS metrics available."
    )


st.divider()


# ============================================================
# QUESTION LEVEL RESULTS
# ============================================================

st.subheader(
    "Question-Level Evaluation"
)


if not questions:

    st.warning(
        "No question-level results found."
    )

else:

    rows = []

    for item in questions:

        rows.append(

            {

                "Question":
                    item.get(
                        "question",
                        "",
                    ),

                "Route":
                    item.get(
                        "route",
                        "",
                    ),

                "Status":
                    item.get(
                        "ragas_status",
                        "",
                    ),

                "Faithfulness":
                    item.get(
                        "faithfulness"
                    ),

                "Answer Relevancy":
                    item.get(
                        "answer_relevancy"
                    ),

                "Context Precision":
                    item.get(
                        "context_precision"
                    ),

                "Context Recall":
                    item.get(
                        "context_recall"
                    ),

                "Reason":
                    item.get(
                        "reason",
                        "",
                    ),

            }

        )

    question_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        question_df,
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# ============================================================
# RAGAS-ONLY RESULTS
# ============================================================

st.subheader(
    "RAGAS Evaluated Questions"
)


evaluated_questions = [

    item

    for item in questions

    if item.get(
        "ragas_status"
    ) == "evaluated"

]


if evaluated_questions:

    evaluated_df = pd.DataFrame(
        [

            {

                "Question":
                    item.get(
                        "question"
                    ),

                "Faithfulness":
                    item.get(
                        "faithfulness"
                    ),

                "Answer Relevancy":
                    item.get(
                        "answer_relevancy"
                    ),

                "Context Precision":
                    item.get(
                        "context_precision"
                    ),

                "Context Recall":
                    item.get(
                        "context_recall"
                    ),

            }

            for item
            in evaluated_questions

        ]
    )

    st.dataframe(
        evaluated_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No questions were evaluated by RAGAS."
    )


st.divider()


# ============================================================
# ROUTING ANALYSIS
# ============================================================

st.subheader(
    "Agent Routing Analysis"
)


route_counts = {}


for item in questions:

    route = item.get(
        "route",
        "unknown",
    )

    route_counts[route] = (
        route_counts.get(
            route,
            0,
        )
        + 1
    )


if route_counts:

    route_df = pd.DataFrame(
        {
            "Route":
                list(
                    route_counts.keys()
                ),

            "Questions":
                list(
                    route_counts.values()
                ),
        }
    )

    st.bar_chart(
        route_df.set_index(
            "Route"
        )
    )


st.divider()


# ============================================================
# SKIPPED QUESTIONS
# ============================================================

st.subheader(
    "RAGAS Skipped Questions"
)


skipped_questions = [

    item

    for item in questions

    if item.get(
        "ragas_status"
    ) == "skipped"

]


if skipped_questions:

    skipped_df = pd.DataFrame(
        [

            {

                "Question":
                    item.get(
                        "question"
                    ),

                "Route":
                    item.get(
                        "route"
                    ),

                "Reason":
                    item.get(
                        "reason",
                        "",
                    ),

            }

            for item
            in skipped_questions

        ]
    )

    st.dataframe(
        skipped_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "No questions were skipped."
    )


st.divider()


# ============================================================
# EVALUATION ERRORS
# ============================================================

st.subheader(
    "Evaluation Errors"
)


error_questions = [

    item

    for item in questions

    if item.get(
        "ragas_status"
    ) == "error"

]


if error_questions:

    for item in error_questions:

        st.error(
            f"Question: "
            f"{item.get('question')}\n\n"
            f"Error: "
            f"{item.get('reason', 'Unknown error')}"
        )

else:

    st.success(
        "✅ No RAGAS evaluation errors."
    )


# ============================================================
# ============================================================
# MODULE 12 — LLM OBSERVABILITY
# ============================================================
# ============================================================

st.divider()

st.header(
    "🔭 LLM Observability"
)

st.caption(
    "Live operational metrics collected from LangSmith"
)


# ============================================================
# LOAD LANGSMITH DATA
# ============================================================

try:

    observability_data = get_observability_data(
        hours=observation_hours,
        limit=observation_limit,
    )

    observability_summary = calculate_summary(
        observability_data
    )

except Exception as exc:

    observability_data = []

    observability_summary = {}

    st.error(
        "Could not load LangSmith observability data."
    )

    st.code(
        str(exc)
    )


# ============================================================
# CHECK OBSERVABILITY DATA
# ============================================================

if not observability_data:

    st.warning(
        "No LangSmith runs were found for the selected time window."
    )

    st.info(
        "Make sure LangSmith tracing is enabled and your application "
        "has generated requests."
    )

else:

    # ========================================================
    # REQUEST METRICS
    # ========================================================

    st.subheader(
        "Request Health"
    )

    obs_col1, obs_col2, obs_col3, obs_col4 = (
        st.columns(4)
    )


    with obs_col1:

        st.metric(
            "Total Requests",
            observability_summary.get(
                "total_requests",
                0,
            ),
        )


    with obs_col2:

        st.metric(
            "Successful",
            observability_summary.get(
                "successful_requests",
                0,
            ),
        )


    with obs_col3:

        st.metric(
            "Failed",
            observability_summary.get(
                "failed_requests",
                0,
            ),
        )


    with obs_col4:

        error_rate = (
            observability_summary.get(
                "error_rate",
                0,
            )
            * 100
        )

        st.metric(
            "Error Rate",
            f"{error_rate:.2f}%",
        )


    # ========================================================
    # LATENCY
    # ========================================================

    st.subheader(
        "Latency"
    )


    latency_col1, latency_col2, latency_col3 = (
        st.columns(3)
    )


    with latency_col1:

        st.metric(
            "Average Latency",
            f"{observability_summary.get('avg_latency', 0):.2f} s",
        )


    with latency_col2:

        st.metric(
            "P50 Latency",
            f"{observability_summary.get('p50_latency', 0):.2f} s",
        )


    with latency_col3:

        st.metric(
            "P95 Latency",
            f"{observability_summary.get('p95_latency', 0):.2f} s",
        )


    # ========================================================
    # COST
    # ========================================================

    st.subheader(
        "LLM Cost"
    )


    cost_col1, cost_col2, cost_col3 = (
        st.columns(3)
    )


    with cost_col1:

        st.metric(
            "Total Cost",
            f"${observability_summary.get('total_cost', 0):.6f}",
        )


    with cost_col2:

        st.metric(
            "Input Cost",
            f"${observability_summary.get('input_cost', 0):.6f}",
        )


    with cost_col3:

        st.metric(
            "Output Cost",
            f"${observability_summary.get('output_cost', 0):.6f}",
        )


    # ========================================================
    # TOKEN USAGE
    # ========================================================

    st.subheader(
        "Token Usage"
    )


    token_col1, token_col2, token_col3 = (
        st.columns(3)
    )


    with token_col1:

        st.metric(
            "Input Tokens",
            f"{observability_summary.get('input_tokens', 0):,}",
        )


    with token_col2:

        st.metric(
            "Output Tokens",
            f"{observability_summary.get('output_tokens', 0):,}",
        )


    with token_col3:

        st.metric(
            "Total Tokens",
            f"{observability_summary.get('total_tokens', 0):,}",
        )


    if (
        observability_summary.get(
            "total_tokens",
            0,
        )
        == 0
    ):

        st.info(
            "Token usage is not available for these LangSmith "
            "run records. Cost information is available."
        )


    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    observability_df = pd.DataFrame(
        observability_data
    )


    # ========================================================
    # COST BY RUN
    # ========================================================

    st.subheader(
        "Cost by Run"
    )


    if (
        "total_cost"
        in observability_df.columns
    ):

        cost_chart_df = (
            observability_df[
                [
                    "start_time",
                    "total_cost",
                ]
            ]
            .copy()
        )

        cost_chart_df = (
            cost_chart_df
            .dropna(
                subset=[
                    "start_time"
                ]
            )
        )

        if not cost_chart_df.empty:

            cost_chart_df[
                "start_time"
            ] = pd.to_datetime(
                cost_chart_df[
                    "start_time"
                ]
            )

            cost_chart_df = (
                cost_chart_df
                .sort_values(
                    "start_time"
                )
                .set_index(
                    "start_time"
                )
            )

            st.line_chart(
                cost_chart_df[
                    "total_cost"
                ]
            )


    # ========================================================
    # LATENCY BY RUN
    # ========================================================

    st.subheader(
        "Latency by Run"
    )


    if (
        "latency"
        in observability_df.columns
    ):

        latency_chart_df = (
            observability_df[
                [
                    "start_time",
                    "latency",
                ]
            ]
            .dropna(
                subset=[
                    "start_time",
                    "latency",
                ]
            )
        )

        if not latency_chart_df.empty:

            latency_chart_df[
                "start_time"
            ] = pd.to_datetime(
                latency_chart_df[
                    "start_time"
                ]
            )

            latency_chart_df = (
                latency_chart_df
                .sort_values(
                    "start_time"
                )
                .set_index(
                    "start_time"
                )
            )

            st.line_chart(
                latency_chart_df[
                    "latency"
                ]
            )


    # ========================================================
    # ROUTE DISTRIBUTION
    # ========================================================

    st.subheader(
        "Route Distribution"
    )


    route_observability = (
        observability_df[
            "route"
        ]
        .fillna(
            "unknown"
        )
        .value_counts()
    )


    if not route_observability.empty:

        route_obs_df = pd.DataFrame(
            {
                "Route":
                    route_observability.index,

                "Requests":
                    route_observability.values,
            }
        )

        st.bar_chart(
            route_obs_df.set_index(
                "Route"
            )
        )


    # ========================================================
    # ERROR MONITORING
    # ========================================================

    st.subheader(
        "Runtime Errors"
    )


    runtime_errors = observability_df[
        observability_df[
            "status"
        ] == "error"
    ]


    if runtime_errors.empty:

        st.success(
            "✅ No runtime errors found."
        )

    else:

        st.warning(
            f"{len(runtime_errors)} runtime errors "
            f"were detected."
        )

        error_columns = [

            "start_time",
            "run_type",
            "question",
            "error",

        ]

        available_error_columns = [

            column

            for column
            in error_columns

            if column
            in runtime_errors.columns

        ]

        st.dataframe(
            runtime_errors[
                available_error_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # RECENT RUNS
    # ========================================================

    st.subheader(
        "Recent LangSmith Runs"
    )


    recent_columns = [

        "start_time",
        "run_type",
        "status",
        "route",
        "latency",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "total_cost",
    ]


    available_columns = [

        column

        for column
        in recent_columns

        if column
        in observability_df.columns

    ]


    recent_df = (
        observability_df[
            available_columns
        ]
        .copy()
    )


    if (
        "start_time"
        in recent_df.columns
    ):

        recent_df = (
            recent_df
            .sort_values(
                "start_time",
                ascending=False,
            )
            .head(20)
        )


    st.dataframe(
        recent_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Enterprise Agentic RAG | "
    "LLMOps | "
    "RAGAS 0.4.3 | "
    "LangSmith Observability"
)