from langchain_groq import ChatGroq
from app.config.settings import settings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


# -------------------------
# Evaluator LLM
# -------------------------

groq_llm = ChatGroq(
    model=settings.MODEL_NAME,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

evaluator_llm = LangchainLLMWrapper(groq_llm)


# -------------------------
# Evaluator Embeddings
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    embeddings
)


# -------------------------
# Configure metrics
# -------------------------

faithfulness.llm = evaluator_llm

answer_relevancy.llm = evaluator_llm
answer_relevancy.embeddings = evaluator_embeddings

context_precision.llm = evaluator_llm

context_recall.llm = evaluator_llm


# -------------------------
# Evaluate ONE example
# -------------------------

def evaluate_rag_sample(
    question,
    answer,
    contexts,
    ground_truth,
):

    row = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }

    scores = {}

    scores["faithfulness"] = faithfulness.score(row)

    scores["answer_relevancy"] = answer_relevancy.score(row)

    scores["context_precision"] = context_precision.score(row)

    scores["context_recall"] = context_recall.score(row)

    return scores