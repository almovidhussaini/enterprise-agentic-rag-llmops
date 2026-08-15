from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.graph import graph


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Enterprise Agentic RAG",
    description="Enterprise Agentic RAG + LLMOps API",
    version="1.0",
)


# ============================================================
# Request Schema
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "project": "Enterprise Agentic RAG + LLMOps",
        "version": "1.0",
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Ask Endpoint
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        result = graph.invoke({
            "question": request.question
        })

        return {
            "question": request.question,
            "route": result.get("route"),
            "answer": result.get("answer", ""),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )