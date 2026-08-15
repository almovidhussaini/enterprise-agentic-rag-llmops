from app.evaluation.ragas_eval import evaluate_rag_sample


def main():

    question = "What is multi-head attention?"

    answer = (
        "Multi-head attention projects queries, keys and values into "
        "multiple representation subspaces and computes attention "
        "independently in each subspace."
    )

    contexts = [
        (
            "Multi-head attention allows the model to jointly attend "
            "to information from different representation subspaces."
        )
    ]

    ground_truth = (
        "Multi-head attention uses multiple attention heads to capture "
        "different relationships in the input."
    )

    result = evaluate_rag_sample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    print("\n==============================")
    print("RAGAS RESULT")
    print("==============================")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()