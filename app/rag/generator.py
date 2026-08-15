from langchain_groq import ChatGroq
from langchain_core.tracers import LangChainTracer
from app.config.settings import settings
from langsmith import traceable


def get_llm():

    tracer = LangChainTracer(
        project_name=
        "enterprise-agentic-rag"
    )

    llm = ChatGroq(

        model="llama-3.3-70b-versatile",

        temperature=0,

        api_key=settings.GROQ_API_KEY,

        callbacks=[
            tracer
        ]
    )


    return llm


@traceable(
    name="RAG Generation",
    run_type="llm"
)
def generate_answer(question, context):

    prompt = f"""
You are a research paper assistant.

Answer the question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- If the answer is not available in the context, say:
  "The information is not available in the provided paper."
- Mention when information comes from the document.

Context:
------------------
{context}
------------------

Question:
{question}

Answer:
"""
    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content