ROUTER_PROMPT = """
You are a routing agent.

Classify the user's question into exactly one category.

Return ONLY one word.

general
vector_rag
graph_rag

Use vector_rag if the question:

- refers to an uploaded document
- asks to explain something likely contained in the uploaded document
- asks about a paper, PDF, report, article, research, chapter, or document
- asks for specific facts that should come from retrieved context

Use graph_rag if the question:

- compares multiple topics
- asks for relationships
- requests a global summary
- asks to connect concepts

Otherwise return general.

Question:
{question}

Answer:
"""