from app.rag.retriever import retrieve_documents

from app.rag.generator import generate_answer



def rag_answer(question):


    documents = retrieve_documents(
        question
    )


    context = "\n\n".join(

        [
            doc.page_content
            for doc in documents
        ]

    )


    answer = generate_answer(

        question,

        context

    )


    return {

        "question": question,

        "answer": answer,

        "sources": [

            doc.metadata

            for doc in documents

        ]

    }