from app.rag.retriever import retrieve_documents
from app.rag.generator import generate_answer
from app.rag.generator import get_llm
from app.services.router_service import classify_question
from langsmith import traceable

def router_node(state):

    route = classify_question(

        state["question"]

    )

    state["route"] = route

    return state

def retrieval_node(state):

    docs = retrieve_documents(
        state["question"]
    )

    print("Number of docs:", len(docs))

    # for i, doc in enumerate(docs):
    #     print("\n========== DOC", i, "==========")
    #     print(doc.page_content[:1000])

    state["retrieved_docs"] = docs

    return state


def rag_generation_node(state):

    context = "\n\n".join(
        doc.page_content
        for doc in state["retrieved_docs"]
    )

    answer = generate_answer(
        state["question"],
        context
    )

    state["answer"] = answer

    return state

def llm_generation_node(state):

    llm = get_llm()

    response = llm.invoke(

        state["question"]

    )

    # print(response.content,'responce content in nodes')
    state["answer"] = response.content

    return state

def router_edge(state):

    return state["route"]



def retrieval_decision(state):

    return state["retrieval_status"]



def retrieval_grader_node(state):

    docs = state["retrieved_docs"]

    if len(docs) == 0:
        state["retrieval_status"] = "rewrite"

    else:
        state["retrieval_status"] = "generate"

    return state


def rewrite_query_node(state):

    llm = get_llm()

    prompt = f"""

You are a research paper retrieval assistant.

Rewrite the question so that it retrieves relevant sections
from an academic paper.

Include possible keywords from the paper:

- architecture
- methodology
- model design
- training procedure
- experiments
- evaluation
- datasets

Original question:

{state["question"]}

Return only the rewritten search query.

"""

    response = llm.invoke(prompt)

    print("Rewritten Query:", response.content)

    state["question"] = response.content.strip()

    return state