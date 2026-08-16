# Enterprise Agentic RAG + LLMOps

An end-to-end **Agentic RAG and LLMOps learning project** demonstrating document retrieval, agentic routing, evaluation, observability, CI/CD, Docker, and an evaluation dashboard.

## Architecture

```text
                         User
                          │
                          ▼
                   Streamlit UI
                          │
                          ▼
                    FastAPI API
                          │
                          ▼
                    LangGraph
                   Agentic Router
                    /          \
                   /            \
                  ▼              ▼
             RAG Route       General Route
                │
        ┌───────┴────────┐
        ▼                ▼
     FAISS              BM25
        │                │
        └───────┬────────┘
                ▼
             Reranker
                │
                ▼
              LLM
             (Groq)
                │
                ▼
             Response

        ─────────────────────
              LLMOps
        ─────────────────────
        │        │        │
        ▼        ▼        ▼
    LangSmith  RAGAS   Dashboard
    Tracing   Eval.    Streamlit
        │
        ▼
   GitHub Actions
        │
        ▼
   Docker / GHCR


Main Features
PDF document ingestion and processing
Semantic/text chunking
BGE embeddings
FAISS vector search
BM25 keyword retrieval
BGE reranking
LangGraph agentic routing
General question handling
Groq LLM integration
FastAPI backend
Streamlit frontend
RAGAS evaluation
LangSmith tracing and monitoring
Evaluation dashboard
Docker containerization
GitHub Actions CI/CD
GitHub Container Registry (GHCR)
Project Structure
enterprise-agentic-rag-llmops/
│
├── app/
│   ├── agents/
│   ├── api/
│   ├── evaluation/
│   ├── retrieval/
│   └── ...
│
├── data/
│   ├── vectorstore/
│   └── evaluation_results.json
│
├── evaluation/
│   └── dataset.json
│
├── streamlit_app.py
├── Dockerfile
├── docker-compose.prod.yml
├── requirements.txt
├── requirements-api.txt
└── .github/
    └── workflows/
        ├── ci.yml
        ├── deploy.yml
        └── rollback.yml
Run Locally
1. Clone the repository
git clone https://github.com/almovidhussaini/enterprise-agentic-rag-llmops.git
cd enterprise-agentic-rag-llmops
2. Create virtual environment
python -m venv venv

Windows:

venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=enterprise-agentic-rag
5. Start FastAPI
uvicorn app.api.main:app --reload

API:

http://localhost:8000

Health check:

http://localhost:8000/health
6. Start Streamlit

In another terminal:

streamlit run streamlit_app.py

Dashboard:

http://localhost:8501
API
Health
GET /health
Ask Question
POST /ask

Example:

{
  "question": "What is positional encoding in the Transformer?"
}

Response:

{
  "question": "What is positional encoding in the Transformer?",
  "route": "vector_rag",
  "answer": "...",
  "latency_seconds": 2.31,
  "retrieved_documents": 4
}

For general questions, the agent can use a non-RAG route where no documents are retrieved.

Evaluation

The project uses RAGAS for evaluating:

Faithfulness
Answer Relevancy
Context Precision
Context Recall

Evaluation results are stored in:

data/evaluation_results.json

The Streamlit dashboard visualizes the evaluation results.

LLMOps

The project demonstrates:

Application
    │
    ├── LangSmith Tracing
    │
    ├── RAGAS Evaluation
    │
    ├── Evaluation Dataset
    │
    ├── Monitoring
    │
    └── Streamlit Dashboard
CI/CD

GitHub Actions automatically:

Runs Python tests
Builds the Docker image
Pushes the image to GHCR
Provides a deployment workflow
Supports rollback workflow

Docker image:

ghcr.io/almovidhussaini/enterprise-agentic-rag-llmops:latest
Docker

The application uses a CPU-based Python Docker image.

The Docker image is built and published through GitHub Actions rather than relying on local Docker builds.

Demo

For demonstration purposes, the backend and Streamlit dashboard can be run locally:

FastAPI  → http://localhost:8000
Streamlit → http://localhost:8501
Learning Goals

This project was built to gain practical experience with:

Agentic AI
RAG
LangGraph
LLMOps
RAGAS
LangSmith
FastAPI
Streamlit
Docker
GitHub Actions
CI/CD
Evaluation and monitoring
Author

Almoveed Hussaini

MS Artificial Intelligence
Bahria University Islamabad