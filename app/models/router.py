from pydantic import BaseModel
from typing import Literal


class RouteDecision(BaseModel):
    route: Literal[
        "general",
        "vector_rag",
        "graph_rag"
    ]