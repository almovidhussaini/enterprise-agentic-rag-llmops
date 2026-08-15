import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

RESULTS_FILE = Path(
    "data/evaluation_results.json"
)


# ============================================================
# GROQ API KEY
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:

    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Please check your .env file."
    )


# ============================================================
# RAGAS EVALUATOR LLM
# ============================================================

groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    max_retries=2,
    api_key=GROQ_API_KEY,
)


# ============================================================
# WRAP GROQ FOR RAGAS
# ============================================================

ragas_llm = LangchainLLMWrapper(
    groq_llm
)


# ============================================================
# RAGAS EMBEDDINGS
# ============================================================

hf_embeddings = HuggingFaceEmbeddings(

    model_name="BAAI/bge-small-en-v1.5",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    },
)


# ============================================================
# WRAP EMBEDDINGS FOR RAGAS
# ============================================================

ragas_embeddings = LangchainEmbeddingsWrapper(
    hf_embeddings
)


# ============================================================
# RAGAS METRICS
# ============================================================

faithfulness_metric = Faithfulness(
    llm=ragas_llm
)


answer_relevancy_metric = AnswerRelevancy(
    llm=ragas_llm,
    embeddings=ragas_embeddings
)


context_precision_metric = ContextPrecision(
    llm=ragas_llm
)


context_recall_metric = ContextRecall(
    llm=ragas_llm
)


# ============================================================
# EVALUATE ONE RAG SAMPLE
# ============================================================

def evaluate_rag_sample(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
):
    """
    Evaluate one RAG question using RAGAS.

    RAGAS uses:

        Groq LLM
            +
        BGE embeddings

    No OpenAI API is required.
    """

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not contexts:

        print(
            "No contexts found."
        )

        print(
            "Skipping RAGAS."
        )

        return None


    # --------------------------------------------------------
    # Build RAGAS dataset
    # --------------------------------------------------------

    data = {

        "user_input": [
            question
        ],

        "response": [
            answer
        ],

        "retrieved_contexts": [
            contexts
        ],

        "reference": [
            ground_truth
        ],
    }


    dataset = Dataset.from_dict(
        data
    )


    # --------------------------------------------------------
    # Run RAGAS
    # --------------------------------------------------------

    result = evaluate(

        dataset=dataset,

        metrics=[

            faithfulness_metric,

            answer_relevancy_metric,

            context_precision_metric,

            context_recall_metric,

        ],

    )


    return result


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

def save_evaluation_results(
    experiment_name,
    dataset_name,
    question_results,
):
    """
    Save evaluation results for the Streamlit
    LLMOps dashboard.
    """

    metric_names = [

        "faithfulness",

        "answer_relevancy",

        "context_precision",

        "context_recall",

    ]


    # --------------------------------------------------------
    # Successfully evaluated questions
    # --------------------------------------------------------

    evaluated_questions = [

        item

        for item in question_results

        if item.get(
            "ragas_status"
        ) == "evaluated"
    ]


    # --------------------------------------------------------
    # Calculate overall metrics
    # --------------------------------------------------------

    metrics = {}


    for metric_name in metric_names:

        values = [

            item.get(
                metric_name
            )

            for item in evaluated_questions

            if item.get(
                metric_name
            ) is not None
        ]


        if values:

            metrics[
                metric_name
            ] = (
                sum(values)
                / len(values)
            )

        else:

            metrics[
                metric_name
            ] = None


    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    output = {

        "experiment":
            experiment_name,

        "dataset":
            dataset_name,

        "timestamp":
            datetime.now().isoformat(),

        "metrics":
            metrics,

        "questions":
            question_results,
    }


    # --------------------------------------------------------
    # Create data directory
    # --------------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

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

    print(
        "Questions:",
        len(question_results)
    )

    print()

    print(
        "OVERALL METRICS"
    )

    print("-" * 70)


    for metric_name in metric_names:

        value = metrics.get(
            metric_name
        )


        if value is None:

            display_value = "N/A"

        else:

            display_value = (
                f"{value:.4f}"
            )


        print(
            f"{metric_name:<25}"
            f"{display_value}"
        )


    print("=" * 70)


    return output


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "RAGAS Evaluation Module"
    )

    print("=" * 60)

    print()

    print(
        "This module provides:"
    )

    print(
        "  - evaluate_rag_sample()"
    )

    print(
        "  - save_evaluation_results()"
    )

    print()

    print(
        "Evaluator LLM: Groq"
    )

    print(
        "Model: llama-3.1-8b-instant"
    )

    print(
        "Evaluator Embeddings:"
    )

    print(
        "BAAI/bge-small-en-v1.5"
    )

    print()

    print(
        "No evaluation was executed directly."
    )

    print(
        "The functions should be called "
        "by your evaluation pipeline."
    )