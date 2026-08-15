from app.rag.loader import load_pdf

from app.rag.splitter import split_documents

from app.rag.embeddings import get_embedding_model

from app.rag.vector_store import (
    create_vector_store,
    save_vector_store
)



PDF_PATH = (
    "data/documents/sample.pdf"
)


VECTOR_PATH = (
    "data/vectorstore"
)



def ingest():


    print("Loading PDF...")


    documents = load_pdf(
        PDF_PATH
    )


    print(
        f"Pages: {len(documents)}"
    )



    print("Splitting text...")


    chunks = split_documents(
        documents
    )


    print(
        f"Chunks: {len(chunks)}"
    )



    print("Creating embeddings...")


    embeddings = get_embedding_model()



    print("Creating FAISS DB...")


    vector_store = create_vector_store(

        chunks,

        embeddings

    )



    save_vector_store(

        vector_store,

        VECTOR_PATH

    )


    print(
        "Vector database created!"
    )



if __name__=="__main__":

    ingest()