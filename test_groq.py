from langchain_groq import ChatGroq

from app.config.settings import settings



llm = ChatGroq(
    model=settings.MODEL_NAME,
    api_key=settings.GROQ_API_KEY
)



response = llm.invoke(
    "Explain RAG in one sentence"
)


# print(response.content)