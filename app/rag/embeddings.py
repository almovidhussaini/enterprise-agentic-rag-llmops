from langchain_huggingface import HuggingFaceEmbeddings

_embedding_model = None

def get_embedding_model():
    global _embedding_model

    print("Embedding object:", id(_embedding_model))

    if _embedding_model is None:
        print("Loading embedding model...")

        _embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Created object:", id(_embedding_model))

    return _embedding_model