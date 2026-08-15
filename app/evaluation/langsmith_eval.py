import json
import time
from pathlib import Path

from dotenv import load_dotenv

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

from app.agents.graph import graph
from app.evaluation.ragas_eval import evaluate_rag_sample


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "Transformer RAG Evaluation"

EXPERIMENT_PREFIX = "enterprise-rag-test"

RESULTS_FILE = Path(
    "data/evaluation_results.json"
)


RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


# ============================================================
# LANGSMITH CLIENT
# ============================================================

client = Client()


# ============================================================
# RESULTS
# ============================================================

evaluation_results = []


# ============================================================
# 1. RUN APPLICATION
# ============================================================

def target_function(inputs: dict) -> dict:

    question = inputs["question"]

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------------
    # Run LangGraph
    # --------------------------------------------------------

    result = graph.invoke(
        {
            "question": question
        }
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    route = result.get(
        "route"
    )

    answer = result.get(
        "answer",
        ""
    )

    retrieved_docs = result.get(
        "retrieved_docs",
        []
    )

    # --------------------------------------------------------
    # Documents → strings
    # --------------------------------------------------------

    contexts = []

    for doc in retrieved_docs:

        if hasattr(
            doc,
            "page_content"
        ):

            contexts.append(
                doc.page_content
            )

        elif isinstance(
            doc,
            str
        ):

            contexts.append(
                doc
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("ROUTE:")
    print(route)

    print()
    print("RETRIEVED DOCUMENTS:")
    print(len(contexts))

    print()
    print("ANSWER:")
    print(answer)

    # --------------------------------------------------------
    # Return to LangSmith
    # --------------------------------------------------------

    return {
        "answer": answer,
        "route": route,
        "contexts": contexts,
    }


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset_examples():

    examples = list(
        client.list_examples(
            dataset_name=DATASET_NAME
        )
    )

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(
        "Questions:",
        len(examples)
    )

    return examples


# ============================================================
# 3. RUN RAGAS OUTSIDE LANGSMITH
# ============================================================

def evaluate_with_ragas(
    question,
    ground_truth,
    application_result,
):

    route = application_result.get(
        "route"
    )

    answer = application_result.get(
        "answer",
        ""
    )

    contexts = application_result.get(
        "contexts",
        []
    )

    question_result = {

        "question":
            question,

        "route":
            route,

    }

    # ========================================================
    # GENERAL ROUTE
    # ========================================================

    if route != "vector_rag":

        print()
        print(
            "RAGAS SKIPPED"
        )

        print(
            "Reason: Non-RAG route"
        )

        question_result.update({

            "ragas_status":
                "skipped",

            "reason":
                "Non-RAG route",

        })

        return question_result

    # ========================================================
    # NO CONTEXT
    # ========================================================

    if not contexts:

        print()
        print(
            "RAGAS SKIPPED"
        )

        print(
            "Reason: No contexts"
        )

        question_result.update({

            "ragas_status":
                "skipped",

            "reason":
                "No retrieved contexts",

        })

        return question_result

    # ========================================================
    # RUN RAGAS
    # ========================================================

    print()
    print("-" * 70)
    print("RUNNING RAGAS")
    print("-" * 70)

    print(
        "Question:",
        question
    )

    print(
        "Contexts:",
        len(contexts)
    )

    try:

        # ----------------------------------------------------
        # IMPORTANT
        #
        # RAGAS is now completely outside LangSmith evaluator.
        # ----------------------------------------------------

        result = evaluate_rag_sample(

            question=question,

            answer=answer,

            contexts=contexts,

            ground_truth=ground_truth,

        )

        if result is None:

            question_result.update({

                "ragas_status":
                    "skipped",

                "reason":
                    "RAGAS returned None",

            })

            return question_result

        # ====================================================
        # Convert to DataFrame
        # ====================================================

        dataframe = (
            result.to_pandas()
        )

        print()
        print(
            "RAGAS RESULT:"
        )

        print(
            dataframe
        )

        if dataframe.empty:

            question_result.update({

                "ragas_status":
                    "error",

                "reason":
                    "Empty RAGAS result",

            })

            return question_result

        # ====================================================
        # FIRST ROW
        # ====================================================

        row = dataframe.iloc[0]

        # ====================================================
        # METRICS
        # ====================================================

        scores = {}

        for metric in RAGAS_METRICS:

            value = row.get(
                metric
            )

            try:

                scores[metric] = (
                    float(value)
                    if value is not None
                    else None
                )

            except (
                TypeError,
                ValueError,
            ):

                scores[metric] = None

        # ====================================================
        # DISPLAY
        # ====================================================

        print()
        print(
            "RAGAS SCORES:"
        )

        print(
            scores
        )

        # ====================================================
        # SAVE QUESTION RESULT
        # ====================================================

        question_result.update({

            "ragas_status":
                "evaluated",

            "faithfulness":
                scores.get(
                    "faithfulness"
                ),

            "answer_relevancy":
                scores.get(
                    "answer_relevancy"
                ),

            "context_precision":
                scores.get(
                    "context_precision"
                ),

            "context_recall":
                scores.get(
                    "context_recall"
                ),

        })

        return question_result

    except Exception as e:

        print()
        print(
            "RAGAS ERROR:"
        )

        print(
            type(e).__name__
        )

        print(e)

        question_result.update({

            "ragas_status":
                "error",

            "reason":
                str(e),

        })

        return question_result


# ============================================================
# 4. SAVE DASHBOARD RESULTS
# ============================================================

def save_results():

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    metrics = {}

    for metric in RAGAS_METRICS:

        values = []

        for item in evaluation_results:

            if (
                item.get(
                    "ragas_status"
                )
                == "evaluated"
            ):

                value = item.get(
                    metric
                )

                if value is not None:

                    values.append(
                        value
                    )

        if values:

            metrics[metric] = (
                sum(values)
                / len(values)
            )

        else:

            metrics[metric] = None

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "total_questions":
            len(evaluation_results),

        "ragas_evaluated":
            sum(
                1
                for x in evaluation_results
                if x.get(
                    "ragas_status"
                ) == "evaluated"
            ),

        "ragas_skipped":
            sum(
                1
                for x in evaluation_results
                if x.get(
                    "ragas_status"
                ) == "skipped"
            ),

        "ragas_errors":
            sum(
                1
                for x in evaluation_results
                if x.get(
                    "ragas_status"
                ) == "error"
            ),

    }

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

    output = {

        "experiment":
            EXPERIMENT_PREFIX,

        "dataset":
            DATASET_NAME,

        "timestamp":
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),

        "metrics":
            metrics,

        "summary":
            summary,

        "questions":
            evaluation_results,

    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "EVALUATION RESULTS SAVED"
    )
    print("=" * 70)

    print(
        "File:",
        RESULTS_FILE
    )

    print()
    print(
        "OVERALL METRICS"
    )

    print("-" * 70)

    for metric in RAGAS_METRICS:

        value = metrics.get(
            metric
        )

        if value is None:

            print(
                f"{metric:<25} N/A"
            )

        else:

            print(
                f"{metric:<25}"
                f"{value:.4f}"
            )

    print("-" * 70)

    print(
        "Total questions:",
        summary[
            "total_questions"
        ]
    )

    print(
        "RAGAS evaluated:",
        summary[
            "ragas_evaluated"
        ]
    )

    print(
        "RAGAS skipped:",
        summary[
            "ragas_skipped"
        ]
    )

    print(
        "RAGAS errors:",
        summary[
            "ragas_errors"
        ]
    )


# ============================================================
# 5. MAIN
# ============================================================

def main():

    global evaluation_results

    evaluation_results = []

    print()
    print("=" * 70)
    print(
        "STARTING LLMOPS EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        "Dataset:",
        DATASET_NAME
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    examples = (
        load_dataset_examples()
    )

    # ========================================================
    # RUN LANGSMITH EXPERIMENT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STARTING LANGSMITH EXPERIMENT"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # LangSmith receives the application function only.
    #
    # No RAGAS evaluator is attached here.
    # --------------------------------------------------------

    results = ls_evaluate(

        target_function,

        data=DATASET_NAME,

        experiment_prefix=
            EXPERIMENT_PREFIX,

        metadata={

            "project":
                "enterprise-agentic-rag",

            "evaluation_type":
                "LLMOps",

            "model":
                "llama-3.1-8b-instant",

            "embedding_model":
                "BAAI/bge-small-en-v1.5",

        },

    )

    print()
    print("=" * 70)
    print(
        "LANGSMITH EXPERIMENT COMPLETE"
    )
    print("=" * 70)

    print(
        results
    )

    # ========================================================
    # RUN RAGAS SEPARATELY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STARTING RAGAS EVALUATION"
    )
    print("=" * 70)

    # ========================================================
    # IMPORTANT
    #
    # We run each question separately.
    #
    # This is slower than batching but avoids the nested
    # LangSmith/RAGAS tracing problem you encountered.
    # ========================================================

    for index, example in enumerate(
        examples,
        start=1
    ):

        print()
        print("=" * 70)

        print(
            f"RAGAS QUESTION "
            f"{index}/{len(examples)}"
        )

        print("=" * 70)

        inputs = (
            example.inputs
            or {}
        )

        question = inputs.get(
            "question",
            ""
        )

        ground_truth = ""

        if example.outputs:

            ground_truth = (
                example.outputs.get(
                    "ground_truth",
                    ""
                )
            )

        # ----------------------------------------------------
        # Run application directly
        #
        # This is deliberately outside LangSmith's evaluator.
        # ----------------------------------------------------

        application_result = (
            target_function(
                inputs
            )
        )

        # ----------------------------------------------------
        # RAGAS
        # ----------------------------------------------------

        question_result = (
            evaluate_with_ragas(

                question=question,

                ground_truth=ground_truth,

                application_result=
                    application_result,

            )
        )

        evaluation_results.append(
            question_result
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_results()

    print()
    print("=" * 70)
    print(
        "LLMOPS EVALUATION PIPELINE COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()