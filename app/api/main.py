from fastapi import FastAPI


app = FastAPI(
    title="Enterprise Agentic RAG",
    version="1.0"
)



@app.get("/")
def home():

    return {
        "status":"running",
        "project":
        "Enterprise Agentic RAG + LLMOps"
    }