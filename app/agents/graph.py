from langgraph.graph import StateGraph, END
from langchain_core.tracers.context import tracing_v2_enabled
from app.agents.state import AgentState

from app.agents.nodes import (
    router_node,
    router_edge,
    retrieval_node,
    retrieval_grader_node,
    retrieval_decision,
    rewrite_query_node,
    rag_generation_node,
    llm_generation_node,
)

# Create workflow
workflow = StateGraph(AgentState)

# -----------------------------
# Add Nodes
# -----------------------------
workflow.add_node("router", router_node)

workflow.add_node("retriever", retrieval_node)

workflow.add_node("retrieval_grader", retrieval_grader_node)

workflow.add_node("rewrite_query", rewrite_query_node)

workflow.add_node("rag", rag_generation_node)

workflow.add_node("llm", llm_generation_node)

# -----------------------------
# Entry Point
# -----------------------------
workflow.set_entry_point("router")

# -----------------------------
# Router Decision
# -----------------------------
workflow.add_conditional_edges(
    "router",
    router_edge,
    {
        "general": "llm",
        "vector_rag": "retriever",

        # We haven't implemented GraphRAG yet,
        # so temporarily send it to the normal retriever.
        "graph_rag": "retriever",
    },
)

# -----------------------------
# Retrieval Flow
# -----------------------------
workflow.add_edge(
    "retriever",
    "retrieval_grader",
)

workflow.add_conditional_edges(
    "retrieval_grader",
    retrieval_decision,
    {
        "generate": "rag",
        "rewrite": "rewrite_query",
    },
)

workflow.add_edge(
    "rewrite_query",
    "retriever",
)

# -----------------------------
# End Nodes
# -----------------------------
workflow.add_edge(
    "rag",
    END,
)

workflow.add_edge(
    "llm",
    END,
)

# -----------------------------
# Compile
# -----------------------------

with tracing_v2_enabled(
    project_name="enterprise-agentic-rag"
):  
  graph = workflow.compile()