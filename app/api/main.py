import logging
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from app.agents.graph import graph


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("enterprise-rag")


# ============================================================
# Application Metrics
# ============================================================

metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency": 0.0,
    "last_latency": 0.0,
}


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Enterprise Agentic RAG",
    description="Enterprise Agentic RAG + LLMOps API",
    version="1.0",
)

Instrumentator().instrument(app).expose(app)


# ============================================================
# Request Schema
# ============================================================

class QuestionRequest(BaseModel):
    question: str


# ============================================================
# Request Monitoring Middleware
# ============================================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    metrics["total_requests"] += 1

    try:

        response = await call_next(request)

        if response.status_code < 400:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1

        return response

    except Exception:

        metrics["failed_requests"] += 1

        raise

    finally:

        latency = time.perf_counter() - start_time

        metrics["total_latency"] += latency
        metrics["last_latency"] = latency

        logger.info(
            "%s %s | latency=%.4fs",
            request.method,
            request.url.path,
            latency,
        )


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
        "version": "1.0",
    }


# ============================================================
# Metrics Endpoint
# ============================================================

@app.get("/metrics")
def get_metrics():

    total = metrics["total_requests"]

    if total > 0:
        average_latency = metrics["total_latency"] / total
    else:
        average_latency = 0.0

    if total > 0:
        error_rate = (
            metrics["failed_requests"] / total
        )
    else:
        error_rate = 0.0

    return {
        "total_requests": total,
        "successful_requests": metrics["successful_requests"],
        "failed_requests": metrics["failed_requests"],
        "error_rate": round(error_rate, 4),
        "average_latency_seconds": round(
            average_latency,
            4,
        ),
        "last_latency_seconds": round(
            metrics["last_latency"],
            4,
        ),
    }


# ============================================================
# Ask Endpoint
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    logger.info(
        "RAG request received | question=%s",
        request.question,
    )

    try:

        result = graph.invoke(
            {
                "question": request.question
            }
        )

        answer = result.get("answer", "")

        logger.info(
            "RAG request completed | route=%s",
            result.get("route"),
        )

        return {
            "question": request.question,
            "route": result.get("route"),
            "answer": answer,
        }

    except Exception as exc:

        logger.exception(
            "RAG request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
@app.get("/ready")
def readiness():
    """
    Readiness check.

    Indicates whether the application is ready
    to receive requests.
    """

    return {
        "status": "ready",
        "service": "enterprise-agentic-rag",
    }