from app.rag.rag_pipeline import rag_answer



question = """
Explain transformer architecture
"""



result = rag_answer(
    question
)



# print("\nANSWER:\n")

# print(
#     result["answer"]
# )


# print("\nSOURCES:\n")

# print(
#     result["sources"]
# )