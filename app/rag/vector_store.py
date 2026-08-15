from langchain_community.vectorstores import FAISS



def create_vector_store(
        documents,
        embeddings
):


    vector_store = FAISS.from_documents(

        documents,

        embeddings

    )


    return vector_store



def save_vector_store(
        vector_store,
        path
):

    vector_store.save_local(
        path
    )