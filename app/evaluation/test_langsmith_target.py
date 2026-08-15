from app.evaluation.langsmith_eval import rag_target


result = rag_target(
    {
        "question": "According to the uploaded paper, what is the main contribution of the paper?"
    }
)

print("\n" + "=" * 60)
print("TARGET OUTPUT")
print("=" * 60)

print("Answer:")
print(result["answer"])

print("\nRoute:")
print(result["route"])

print("\nNumber of contexts:")
print(len(result["contexts"]))

print("\nFirst context:")
if result["contexts"]:
    print(result["contexts"][0])