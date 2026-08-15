from langsmith import Client

from dotenv import load_dotenv
import os

load_dotenv()
client = Client()


print("LANGSMITH_API_KEY exists:",
      bool(os.getenv("LANGSMITH_API_KEY")))

print("LANGSMITH_PROJECT:",
      os.getenv("LANGSMITH_PROJECT"))

examples = [
    {
        "inputs": {
            "question": "Please explain the encoder discussed in the paper."
        },
        "outputs": {
            "answer": (
                "The encoder consists of a stack of six identical layers. "
                "Each layer contains multi-head self-attention and a "
                "position-wise feed-forward network, with residual connections "
                "and layer normalization."
            )
        }
    },

    {
        "inputs": {
            "question": "What is multi-head attention?"
        },
        "outputs": {
            "answer": (
                "Multi-head attention projects queries, keys and values "
                "using different learned linear projections and performs "
                "attention in parallel across multiple heads."
            )
        }
    },

    {
        "inputs": {
            "question": "What positional encoding is used in the Transformer?"
        },
        "outputs": {
            "answer": (
                "The paper uses sinusoidal positional encoding to inject "
                "information about the position of tokens."
            )
        }
    },

    {
        "inputs": {
            "question": "What datasets were used to evaluate the Transformer?"
        },
        "outputs": {
            "answer": (
                "The Transformer was evaluated on the WMT 2014 "
                "English-to-German and English-to-French translation tasks."
            )
        }
    },

    {
        "inputs": {
            "question": "What is the architecture of the Transformer encoder?"
        },
        "outputs": {
            "answer": (
                "The encoder is composed of six identical layers. "
                "Each layer contains multi-head self-attention followed "
                "by a position-wise feed-forward network."
            )
        }
    }
]


dataset = client.create_dataset(
    dataset_name="Transformer RAG Evaluation",
    description="Evaluation dataset for the Attention Is All You Need RAG system."
)


client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)


print("Dataset created:")
print(dataset.name)
print(dataset.id)