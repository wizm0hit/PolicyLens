"""
CareAssist API
--------------
Thin FastAPI wrapper around the existing RAG pipeline (rag.py).
All RAG logic (retrieval, embeddings, TF-IDF, Gemini) stays untouched.

Run with:
    python api.py
Then open: http://localhost:8000
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import uvicorn


# ---------------------------------------------------------
# Ensure the project root is on sys.path so that
# rag.py / hybrid_retriever.py resolve correctly
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------
# Import the existing RAG pipeline (DO NOT modify rag.py)
# ---------------------------------------------------------

from rag import answer_question          # noqa: E402


# ---------------------------------------------------------
# Count indexed policy documents
# ---------------------------------------------------------

DOCUMENTS_DIR = ROOT / "documents"

def count_documents() -> int:
    """Count PDF files in the documents directory."""
    if DOCUMENTS_DIR.exists():
        return len(list(DOCUMENTS_DIR.glob("*.pdf")))
    return 0


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="CareAssist API",
    description="Hospital Policy Knowledge Assistant",
    version="1.0.0",
)


from typing import Optional, List

# ---------------------------------------------------------
# CORS — allow frontend origin during development & production
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The policy question to answer",
    )


class SourceResult(BaseModel):
    chunk_id: str
    document: Optional[str] = None
    policy_id: Optional[str] = None
    section: Optional[str] = None
    pages: List = Field(default_factory=list)
    text: str
    tfidf_score: float
    embedding_score: float
    hybrid_score: float


class AskResponse(BaseModel):
    answer: str
    sources: List = Field(default_factory=list)
    insufficient_evidence: bool


class StatusResponse(BaseModel):
    status: str
    documents_indexed: int
    message: str


# ---------------------------------------------------------
# API routes
# ---------------------------------------------------------

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Return system operational status."""
    doc_count = count_documents()
    return StatusResponse(
        status="operational",
        documents_indexed=doc_count,
        message=f"{doc_count} policy documents indexed and ready.",
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: QuestionRequest):
    """
    Accepts a policy question and returns a grounded answer with sources.
    Delegates entirely to the existing answer_question() in rag.py.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        # Call the existing RAG pipeline — untouched
        response = answer_question(question)
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "CareAssist is temporarily unavailable. "
                "Please try again in a moment."
            ),
        )

    answer = response.get("answer", "")
    sources = response.get("sources", [])

    # Detect insufficient evidence from the standard refusal phrase
    insufficient = (
        not sources
        or "don't have enough information" in answer.lower()
    )

    return AskResponse(
        answer=answer,
        sources=sources,
        insufficient_evidence=insufficient,
    )


# ---------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------

FRONTEND_DIR = ROOT / "frontend"

if FRONTEND_DIR.exists():
    # Mount CSS and JS subdirectories
    css_dir = FRONTEND_DIR / "css"
    if css_dir.exists():
        app.mount(
            "/css",
            StaticFiles(directory=str(css_dir)),
            name="css",
        )

    js_dir = FRONTEND_DIR / "js"
    if js_dir.exists():
        app.mount(
            "/js",
            StaticFiles(directory=str(js_dir)),
            name="js",
        )

    # Serve assets if they exist
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="assets",
        )

    @app.get("/")
    @app.get("/index.html")
    async def serve_frontend():
        """Serve the main frontend application."""
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Frontend not found. Place index.html in frontend/"}


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 60)
    print("  CareAssist — Hospital Policy Knowledge Assistant")
    print("=" * 60)
    print(f"  Documents indexed : {count_documents()}")
    print(f"  API              : http://localhost:{port}/api/ask")
    print(f"  Frontend         : http://localhost:{port}")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
