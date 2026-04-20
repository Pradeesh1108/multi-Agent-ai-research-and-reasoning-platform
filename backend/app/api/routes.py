"""
FastAPI route definitions for the multi-agent platform.

Endpoints:
    POST /query   – Process a user query through the full agent pipeline.
    POST /upload  – Upload a document (PDF / text) into the knowledge base.
    GET  /health  – System health check.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.models import HealthResponse, QueryRequest, QueryResponse, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Lazy references set during app startup (see main.py) ─────────────────────
# These are injected by the lifespan so we avoid circular imports.
_orchestrator = None
_vector_store = None
_doc_processor = None
_upload_dir: str = ""


def configure_routes(orchestrator, vector_store, doc_processor, upload_dir: str) -> None:
    """Called once at startup to inject dependencies into the router."""
    global _orchestrator, _vector_store, _doc_processor, _upload_dir  # noqa: PLW0603
    _orchestrator = orchestrator
    _vector_store = vector_store
    _doc_processor = doc_processor
    _upload_dir = upload_dir


# ── POST /query ──────────────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process a user query",
    description="Runs the full multi-agent pipeline and returns a structured response.",
)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Handle an incoming user query."""
    if _orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is still initialising. Please try again shortly.",
        )

    logger.info("Received query: %s", request.query[:120])
    try:
        response = await _orchestrator.process_query(request.query)
        return response
    except RuntimeError as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query.",
        ) from exc


# ── POST /upload ─────────────────────────────────────────────────────────────

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    description="Upload a PDF or text file to ingest into the knowledge base.",
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Handle document upload and ingestion."""
    if _vector_store is None or _doc_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is still initialising.",
        )

    # Validate file extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # Save the uploaded file to disk
    upload_path = Path(_upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    file_path = upload_path / filename

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Saved uploaded file: %s", file_path)
    except Exception as exc:
        logger.error("Failed to save file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc

    # Process and ingest into FAISS
    try:
        chunks = await _doc_processor.process_file(str(file_path))
        count = await _vector_store.add_documents(chunks)
        logger.info("Ingested %d chunks from %s", count, filename)
        return UploadResponse(
            filename=filename,
            chunks_created=count,
            message=f"Successfully ingested '{filename}' into the knowledge base ({count} chunks).",
        )
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc


# ── GET /health ──────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the status of all system components.",
)
async def health_check() -> HealthResponse:
    """Return system health status."""
    components = {
        "orchestrator": "ready" if _orchestrator is not None else "not_initialised",
        "vector_store": (
            "ready" if _vector_store is not None and _vector_store.is_ready else "empty"
        ),
        "document_processor": "ready" if _doc_processor is not None else "not_initialised",
    }

    overall = "healthy" if _orchestrator is not None else "initialising"

    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )
