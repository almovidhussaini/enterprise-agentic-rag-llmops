from langsmith import Client
from app.config.settings import settings


print("API key loaded:", bool(settings.LANGCHAIN_API_KEY))

client = Client(
    api_key=settings.LANGCHAIN_API_KEY
)

dataset = client.read_dataset(
    dataset_name="Transformer RAG Test"
)

print("DATASET:", dataset.id)

examples = list(
    client.list_examples(
        dataset_id=dataset.id
    )
)

print("EXAMPLES:", len(examples))

for example in examples:
    print("INPUT:", example.inputs)
    print("OUTPUT:", example.outputs)