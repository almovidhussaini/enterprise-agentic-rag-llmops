import json
from pathlib import Path

from dotenv import load_dotenv

from app.agents.graph import graph
from app.evaluation.ragas_eval import evaluate_rag_sample


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path(
    "evaluation/dataset.json"
)


RESULT_FILE = Path(
    "data/single_question_result.json"
)


# ============================================================
# QUESTION INDEX
# ============================================================
#
# 0 = Multi-head attention
# 1 = Positional encoding
# 2 = Main contribution
# 3 = Scaled dot-product attention
# 4 = Encoder
# 5 = Decoder
#
# Change this number when you want to test another question.
#

QUESTION_INDEX = 4


# ============================================================
# LOAD QUESTION
# ============================================================

def load_question():

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    if not dataset:

        raise ValueError(
            "Dataset is empty."
        )

    if QUESTION_INDEX >= len(dataset):

        raise IndexError(
            f"QUESTION_INDEX {QUESTION_INDEX} "
            f"is outside dataset range."
        )

    item = dataset[
        QUESTION_INDEX
    ]

    return (
        item["question"],
        item.get(
            "ground_truth",
            ""
        ),
    )


# ============================================================
# RUN SINGLE QUESTION
# ============================================================

def run_single_question():

    question, ground_truth = (
        load_question()
    )

    print()
    print("=" * 70)
    print("SINGLE QUESTION EVALUATION")
    print("=" * 70)

    print()
    print("Question:")
    print(question)

    print()
    print("Ground Truth:")
    print(ground_truth)

    # ========================================================
    # RUN LANGGRAPH
    # ========================================================

    print()
    print("-" * 70)
    print("RUNNING LANGGRAPH")
    print("-" * 70)

    result = graph.invoke(
        {
            "question": question
        }
    )

    # ========================================================
    # EXTRACT GRAPH RESULT
    # ========================================================

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

    # ========================================================
    # CONVERT DOCUMENTS TO STRINGS
    # ========================================================

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

    # ========================================================
    # DISPLAY GRAPH RESULT
    # ========================================================

    print()
    print("ROUTE:")
    print(route)

    print()
    print("RETRIEVED DOCUMENTS:")
    print(len(contexts))

    print()
    print("ANSWER:")
    print(answer)

    # ========================================================
    # SKIP NON-RAG QUESTIONS
    # ========================================================

    if route != "vector_rag":

        print()
        print("=" * 70)
        print("RAGAS SKIPPED")
        print("=" * 70)

        print(
            "Reason: route is",
            route
        )

        output = {

            "question": question,

            "route": route,

            "answer": answer,

            "retrieved_documents":
                len(contexts),

            "ragas_status":
                "skipped",

            "reason":
                "Non-RAG route",

        }

        save_result(output)

        return

    # ========================================================
    # NO CONTEXT
    # ========================================================

    if not contexts:

        print()
        print("=" * 70)
        print("RAGAS SKIPPED")
        print("=" * 70)

        print(
            "Reason: no retrieved contexts."
        )

        output = {

            "question": question,

            "route": route,

            "answer": answer,

            "retrieved_documents":
                0,

            "ragas_status":
                "skipped",

            "reason":
                "No retrieved contexts",

        }

        save_result(output)

        return

    # ========================================================
    # RUN RAGAS
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING RAGAS")
    print("=" * 70)

    print()
    print(
        "Question:",
        question
    )

    print(
        "Contexts:",
        len(contexts)
    )

    print(
        "Ground Truth:",
        ground_truth
    )

    try:

        ragas_result = (
            evaluate_rag_sample(

                question=question,

                answer=answer,

                contexts=contexts,

                ground_truth=ground_truth,

            )
        )

        # ====================================================
        # RAGAS RESULT
        # ====================================================

        if ragas_result is None:

            output = {

                "question": question,

                "route": route,

                "answer": answer,

                "retrieved_documents":
                    len(contexts),

                "ragas_status":
                    "skipped",

                "reason":
                    "RAGAS returned None",

            }

            save_result(output)

            return

        # ====================================================
        # CONVERT TO DATAFRAME
        # ====================================================

        print()
        print("=" * 70)
        print("RAGAS RESULT")
        print("=" * 70)

        print(ragas_result)

        dataframe = (
            ragas_result.to_pandas()
        )

        print()
        print("RAGAS DATAFRAME:")
        print(dataframe)

        # ====================================================
        # CHECK EMPTY DATAFRAME
        # ====================================================

        if dataframe.empty:

            output = {

                "question": question,

                "route": route,

                "answer": answer,

                "retrieved_documents":
                    len(contexts),

                "ragas_status":
                    "error",

                "reason":
                    "RAGAS returned empty dataframe",

            }

            save_result(output)

            return

        # ====================================================
        # GET FIRST ROW
        # ========================================================

        row = dataframe.iloc[0]

        # ====================================================
        # EXTRACT METRICS
        # ========================================================

        metrics = {}

        for metric in [

            "faithfulness",

            "answer_relevancy",

            "context_precision",

            "context_recall",

        ]:

            value = row.get(
                metric
            )

            try:

                if value is None:

                    metrics[metric] = None

                else:

                    metrics[metric] = float(
                        value
                    )

            except (
                TypeError,
                ValueError,
            ):

                metrics[metric] = None

        # ====================================================
        # DISPLAY METRICS
        # ====================================================

        print()
        print("=" * 70)
        print("RAGAS METRICS")
        print("=" * 70)

        for name, value in metrics.items():

            if value is None:

                print(
                    f"{name:<25} N/A"
                )

            else:

                print(
                    f"{name:<25} "
                    f"{value:.4f}"
                )

        # ====================================================
        # SAVE SUCCESSFUL RESULT
        # ========================================================

        output = {

            "question":
                question,

            "ground_truth":
                ground_truth,

            "route":
                route,

            "answer":
                answer,

            "retrieved_documents":
                len(contexts),

            "ragas_status":
                "evaluated",

            "metrics":
                metrics,

        }

        save_result(output)

    except Exception as e:

        # ====================================================
        # RAGAS ERROR
        # ====================================================

        print()
        print("=" * 70)
        print("RAGAS ERROR")
        print("=" * 70)

        print(
            type(e).__name__
        )

        print(e)

        output = {

            "question":
                question,

            "ground_truth":
                ground_truth,

            "route":
                route,

            "answer":
                answer,

            "retrieved_documents":
                len(contexts),

            "ragas_status":
                "error",

            "reason":
                str(e),

        }

        save_result(output)


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("RESULT SAVED")
    print("=" * 70)

    print(
        "File:",
        RESULT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_single_question()