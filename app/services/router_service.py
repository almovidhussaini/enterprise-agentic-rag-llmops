from app.rag.generator import get_llm
from app.prompts.router_prompt import ROUTER_PROMPT


llm = get_llm()


def classify_question(question: str) -> str:

    prompt = ROUTER_PROMPT.format(
        question=question
    )

    response = llm.invoke(prompt)

    print("LLM Response:", response.content)

    route = response.content.strip().lower()

    if route not in [
        "general",
        "vector_rag",
        "graph_rag"
    ]:
        route = "general"

    return route