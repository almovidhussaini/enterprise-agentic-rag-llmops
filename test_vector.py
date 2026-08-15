from app.rag.embeddings import get_embedding_model

from langchain_community.vectorstores import FAISS



db = FAISS.load_local(

    "data/vectorstore",

    get_embedding_model(),

    allow_dangerous_deserialization=True

)



results = db.similarity_search(

    "What is transformer architecture?",

    k=3

)



for r in results:

    print("----------------")

    print(r.page_content[:300])