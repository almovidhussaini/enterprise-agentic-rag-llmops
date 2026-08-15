from typing import TypedDict, List
from langchain_core.documents import Document


class AgentState(TypedDict):

    question: str

    # Router decision
    route: str

    # Retrieval decision
    retrieval_status: str

    retrieved_docs: list

    answer: str