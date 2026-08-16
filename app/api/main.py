from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timezone
import time
from app.api.metrics import record_request, get_metrics
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
# Root Endpoint
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
        "status": "healthy",
        "service": "enterprise-agentic-rag",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# Ask Endpoint
# ============================================================
@app.post("/ask")
def ask_question(request: QuestionRequest):

    start_time = time.time()

    if not request.question.strip():

        record_request(
            time.time() - start_time,
            error=True
        )

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        result = graph.invoke({
            "question": request.question
        })

        latency = time.time() - start_time

        record_request(latency)

        return {
            "question": request.question,
            "route": result.get("route"),
            "answer": result.get("answer", ""),
            "latency_seconds": round(latency, 4),
        }

    except Exception as exc:

        latency = time.time() - start_time

        record_request(
            latency,
            error=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

@app.get("/metrics")
def metrics():
    return get_metrics()