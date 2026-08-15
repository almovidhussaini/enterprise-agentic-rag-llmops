from langchain_community.vectorstores import FAISS
from langsmith import traceable
from app.rag.embeddings import get_embedding_model



VECTOR_PATH = "data/vectorstore"

_vector_db = None


def get_vector_database():
    global _vector_db

    if _vector_db is None:
        print("Loading FAISS Vector Store...")

        embeddings = get_embedding_model()

        _vector_db = FAISS.load_local(
            VECTOR_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return _vector_db


@traceable(
    name="Vector Retrieval",
    run_type="retriever"
)

def retrieve_documents(query, k=4):
    vector_db = get_vector_database()

    results = vector_db.similarity_search(query, k=k)

    return results