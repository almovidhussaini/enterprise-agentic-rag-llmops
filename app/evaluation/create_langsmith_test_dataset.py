from langsmith import Client
from dotenv import load_dotenv
import os

load_dotenv()

def main():

    client = Client()

    dataset = client.create_dataset(
        dataset_name="Transformer RAG Test"
    )

    client.create_example(
        inputs={
            "question": "What is multi-head attention?"
        },
        outputs={
            "answer": (
                "Multi-head attention projects queries, keys and values "
                "using different learned linear projections and performs "
                "attention in parallel across multiple heads."
            )
        },
        dataset_id=dataset.id
    )

    print("Dataset created:")
    print(dataset.name)
    print(dataset.id)


if __name__ == "__main__":
    main()