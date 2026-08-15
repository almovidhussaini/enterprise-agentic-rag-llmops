import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langsmith import Client


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LANGSMITH
# ============================================================

client = Client()


PROJECT_NAME = os.getenv(
    "LANGCHAIN_PROJECT",
    "enterprise-agentic-rag",
)


# ============================================================
# HELPER: SAFE INTEGER
# ============================================================

def safe_int(value):
    try:
        if value is None:
            return 0

        return int(value)

    except (TypeError, ValueError):
        return 0


# ============================================================
# TOKEN EXTRACTION
# ============================================================

def extract_token_usage(run):
    """
    Extract token usage from LangSmith Run.

    LangSmith/provider integrations can expose
    token information in different locations.
    """

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    # --------------------------------------------------------
    # 1. token_usage
    # --------------------------------------------------------

    token_usage = getattr(
        run,
        "token_usage",
        None,
    )

    if token_usage:

        input_tokens = safe_int(
            token_usage.get(
                "prompt_tokens"
            )
        )

        output_tokens = safe_int(
            token_usage.get(
                "completion_tokens"
            )
        )

        total_tokens = safe_int(
            token_usage.get(
                "total_tokens"
            )
        )

    # --------------------------------------------------------
    # 2. usage_metadata
    # --------------------------------------------------------

    if (
        input_tokens == 0
        and output_tokens == 0
        and total_tokens == 0
    ):

        usage_metadata = getattr(
            run,
            "usage_metadata",
            None,
        )

        if usage_metadata:

            input_tokens = safe_int(
                usage_metadata.get(
                    "input_tokens"
                )
            )

            output_tokens = safe_int(
                usage_metadata.get(
                    "output_tokens"
                )
            )

            total_tokens = safe_int(
                usage_metadata.get(
                    "total_tokens"
                )
            )

    # --------------------------------------------------------
    # 3. extra / metadata
    # --------------------------------------------------------

    extra = getattr(
        run,
        "extra",
        None,
    ) or {}

    metadata = getattr(
        run,
        "metadata",
        None,
    ) or {}

    # Sometimes usage is inside extra
    usage = extra.get(
        "usage"
    ) or extra.get(
        "token_usage"
    )

    if usage:

        if input_tokens == 0:

            input_tokens = safe_int(
                usage.get(
                    "prompt_tokens"
                )
                or usage.get(
                    "input_tokens"
                )
            )

        if output_tokens == 0:

            output_tokens = safe_int(
                usage.get(
                    "completion_tokens"
                )
                or usage.get(
                    "output_tokens"
                )
            )

        if total_tokens == 0:

            total_tokens = safe_int(
                usage.get(
                    "total_tokens"
                )
            )

    # --------------------------------------------------------
    # Calculate total if provider didn't give it
    # --------------------------------------------------------

    if (
        total_tokens == 0
        and (
            input_tokens > 0
            or output_tokens > 0
        )
    ):

        total_tokens = (
            input_tokens
            +
            output_tokens
        )

    return (
        input_tokens,
        output_tokens,
        total_tokens,
    )


# ============================================================
# GET RECENT RUNS
# ============================================================

def get_recent_runs(
    hours=24,
    limit=100,
):

    start_time = (
        datetime.utcnow()
        -
        timedelta(hours=hours)
    )

    runs = list(
        client.list_runs(
            project_name=PROJECT_NAME,
            start_time=start_time,
            limit=limit,
        )
    )

    return runs


# ============================================================
# EXTRACT RUN
# ============================================================

def extract_run_data(run):

    inputs = (
        getattr(
            run,
            "inputs",
            None,
        )
        or {}
    )

    outputs = (
        getattr(
            run,
            "outputs",
            None,
        )
        or {}
    )

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    (
        input_tokens,
        output_tokens,
        total_tokens,
    ) = extract_token_usage(
        run
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency = None

    start_time = getattr(
        run,
        "start_time",
        None,
    )

    end_time = getattr(
        run,
        "end_time",
        None,
    )

    if start_time and end_time:

        latency = (
            end_time
            -
            start_time
        ).total_seconds()

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    route = outputs.get(
        "route"
    )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = inputs.get(
        "question",
        "",
    )

    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    answer = outputs.get(
        "answer",
        "",
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = getattr(
        run,
        "error",
        None,
    )

    # --------------------------------------------------------
    # Run type
    # --------------------------------------------------------

    run_type = getattr(
        run,
        "run_type",
        "",
    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    total_cost = 0.0

    input_cost = 0.0

    output_cost = 0.0

    # LangSmith may expose these directly
    total_cost = getattr(
        run,
        "total_cost",
        None,
    ) or 0.0

    input_cost = getattr(
        run,
        "input_cost",
        None,
    ) or 0.0

    output_cost = getattr(
        run,
        "output_cost",
        None,
    ) or 0.0

    return {

        "id":
            str(
                getattr(
                    run,
                    "id",
                    "",
                )
            ),

        "name":
            getattr(
                run,
                "name",
                "",
            ),

        "run_type":
            run_type,

        "status":
            "error"
            if error
            else "success",

        "question":
            question,

        "route":
            route,

        "answer":
            answer,

        "latency":
            latency,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "total_cost":
            float(
                total_cost
            ),

        "input_cost":
            float(
                input_cost
            ),

        "output_cost":
            float(
                output_cost
            ),

        "error":
            error,

        "start_time":
            start_time,
    }


# ============================================================
# GET OBSERVABILITY DATA
# ============================================================

def get_observability_data(
    hours=24,
    limit=100,
):

    runs = get_recent_runs(
        hours=hours,
        limit=limit,
    )

    data = []

    for run in runs:

        try:

            data.append(
                extract_run_data(
                    run
                )
            )

        except Exception as exc:

            print(
                "Could not process run:",
                exc,
            )

    return data


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values,
    percentage,
):

    if not values:

        return 0.0

    values = sorted(
        values
    )

    index = int(
        len(values)
        *
        percentage
    )

    index = min(
        index,
        len(values) - 1,
    )

    return values[index]


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(data):

    if not data:

        return {

            "total_requests": 0,

            "successful_requests": 0,

            "failed_requests": 0,

            "error_rate": 0.0,

            "avg_latency": 0.0,

            "p50_latency": 0.0,

            "p95_latency": 0.0,

            "input_tokens": 0,

            "output_tokens": 0,

            "total_tokens": 0,

            "total_cost": 0.0,

            "input_cost": 0.0,

            "output_cost": 0.0,
        }

    # --------------------------------------------------------
    # Request statistics
    # --------------------------------------------------------

    total_requests = len(
        data
    )

    successful_requests = sum(

        1

        for item in data

        if item["status"]
        == "success"

    )

    failed_requests = (
        total_requests
        -
        successful_requests
    )

    error_rate = (
        failed_requests
        /
        total_requests
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latencies = [

        item["latency"]

        for item in data

        if item["latency"]
        is not None

    ]

    if latencies:

        avg_latency = (
            sum(latencies)
            /
            len(latencies)
        )

        p50_latency = percentile(
            latencies,
            0.50,
        )

        p95_latency = percentile(
            latencies,
            0.95,
        )

    else:

        avg_latency = 0.0
        p50_latency = 0.0
        p95_latency = 0.0

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    input_tokens = sum(

        item["input_tokens"]

        for item in data

    )

    output_tokens = sum(

        item["output_tokens"]

        for item in data

    )

    total_tokens = sum(

        item["total_tokens"]

        for item in data

    )

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    total_cost = sum(

        item["total_cost"]

        for item in data

    )

    input_cost = sum(

        item["input_cost"]

        for item in data

    )

    output_cost = sum(

        item["output_cost"]

        for item in data

    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "total_requests":
            total_requests,

        "successful_requests":
            successful_requests,

        "failed_requests":
            failed_requests,

        "error_rate":
            error_rate,

        "avg_latency":
            avg_latency,

        "p50_latency":
            p50_latency,

        "p95_latency":
            p95_latency,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "total_cost":
            total_cost,

        "input_cost":
            input_cost,

        "output_cost":
            output_cost,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "LANGSMITH OBSERVABILITY"
    )

    print(
        "=" * 70
    )

    data = get_observability_data(
        hours=24,
        limit=100,
    )

    print()

    print(
        "Runs:",
        len(data),
    )

    summary = calculate_summary(
        data
    )

    print()

    print(
        "SUMMARY"
    )

    print(
        "-" * 70
    )

    for key, value in summary.items():

        print(
            f"{key:<25}: {value}"
        )