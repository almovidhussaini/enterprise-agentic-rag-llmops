import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# Local evaluation dataset
DATASET_FILE = Path(
    "evaluation/dataset.json"
)

# Name of the LangSmith dataset
DATASET_NAME = (
    "Transformer RAG Evaluation"
)


# ============================================================
# LOAD LOCAL DATASET
# ============================================================

def load_local_dataset():
    """
    Load questions and ground-truth answers
    from evaluation/dataset.json.
    """

    print()
    print("=" * 70)
    print("LOADING LOCAL EVALUATION DATASET")
    print("=" * 70)

    print(
        f"File: {DATASET_FILE}"
    )

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{DATASET_FILE}"
        )

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            "dataset.json must contain "
            "a JSON list."
        )

    print(
        f"Questions loaded: {len(data)}"
    )

    return data


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(data):
    """
    Validate that every example contains:

        question
        ground_truth
    """

    print()
    print(
        "Validating dataset..."
    )

    for index, item in enumerate(data):

        if not isinstance(
            item,
            dict
        ):

            raise ValueError(
                f"Example {index + 1} "
                f"is not a JSON object."
            )

        if "question" not in item:

            raise ValueError(
                f"Example {index + 1} "
                f"is missing 'question'."
            )

        if "ground_truth" not in item:

            raise ValueError(
                f"Example {index + 1} "
                f"is missing 'ground_truth'."
            )

        if not item["question"].strip():

            raise ValueError(
                f"Example {index + 1} "
                f"has an empty question."
            )

        if not item["ground_truth"].strip():

            raise ValueError(
                f"Example {index + 1} "
                f"has an empty ground_truth."
            )

    print(
        "Dataset validation successful."
    )


# ============================================================
# CREATE LANGSMITH DATASET
# ============================================================

def create_langsmith_dataset(
    client,
    data,
):
    """
    Create a new LangSmith dataset and upload
    all examples.

    If a dataset with the same name already exists,
    the script stops to prevent accidental duplicates.
    """

    print()
    print("=" * 70)
    print("CREATING LANGSMITH DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Check whether dataset already exists
    # --------------------------------------------------------

    existing_datasets = list(
        client.list_datasets(
            dataset_name=DATASET_NAME
        )
    )

    if existing_datasets:

        print()
        print(
            "A LangSmith dataset with this "
            "name already exists:"
        )

        for dataset in existing_datasets:

            print(
                f"  {dataset.name}"
            )

            print(
                f"  ID: {dataset.id}"
            )

        print()
        print(
            "No new dataset was created."
        )

        print(
            "This prevents accidental duplicates."
        )

        return existing_datasets[0]

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Evaluation dataset for "
            "Enterprise Agentic RAG. "
            "Contains Transformer paper "
            "questions and reference answers."
        ),
    )

    print()
    print(
        "LangSmith dataset created:"
    )

    print(
        f"Name: {dataset.name}"
    )

    print(
        f"ID: {dataset.id}"
    )

    # --------------------------------------------------------
    # Prepare examples
    # --------------------------------------------------------

    inputs = []

    outputs = []

    for item in data:

        inputs.append({
            "question": item[
                "question"
            ]
        })

        outputs.append({
            "ground_truth": item[
                "ground_truth"
            ]
        })

    # --------------------------------------------------------
    # Upload examples
    # --------------------------------------------------------

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )

    print()
    print(
        f"Uploaded {len(data)} examples."
    )

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LANGSMITH DATASET MANAGEMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Create LangSmith client
    # --------------------------------------------------------

    client = Client()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_local_dataset()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        data
    )

    # --------------------------------------------------------
    # Create/upload dataset
    # --------------------------------------------------------

    dataset = create_langsmith_dataset(
        client,
        data,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET MANAGEMENT COMPLETE")
    print("=" * 70)

    print(
        f"Dataset name: {dataset.name}"
    )

    print(
        f"Dataset ID: {dataset.id}"
    )

    print()
    print(
        "Number of local examples:"
    )

    print(
        len(data)
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()