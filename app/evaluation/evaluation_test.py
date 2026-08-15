import json
from pathlib import Path

from app.agents.graph import graph
# from app.evaluation.ragas_eval import evaluate_rag
from app.evaluation.ragas_eval import evaluate_rag_sample


def load_dataset():

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = project_root / "evaluation" / "dataset.json"

    print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag(question):

    result = graph.invoke({
        "question": question
    })

    print("\n===== GRAPH STATE =====")
    print("Keys:", result.keys())
    print("======================")

    answer = result["answer"]
    retrieved_docs = result["retrieved_docs"]

    return answer, retrieved_docs


def main():

    dataset = load_dataset()

    questions = []
    answers = []
    retrieved_contexts = []
    ground_truths = []

    for item in dataset:

        question = item["question"]

        print(f"\nQuestion: {question}")

        answer, retrieved_docs = run_rag(question)

        print(f"Answer: {answer}")

        contexts = []

        for doc in retrieved_docs:

            if hasattr(doc, "page_content"):

                contexts.append(doc.page_content)

            elif isinstance(doc, dict) and "page_content" in doc:

                contexts.append(doc["page_content"])

            else:

                contexts.append(str(doc))

        questions.append(question)
        answers.append(answer)
        retrieved_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])

        print(f"Retrieved contexts: {len(contexts)}")


    score = evaluate_rag_sample(
        questions=questions,
        answers=answers,
        retrieved_contexts=retrieved_contexts,
        ground_truths=ground_truths
    )

    print("\n===== RAGAS RESULTS =====")
    print(score)


if __name__ == "__main__":
    main()