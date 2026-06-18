"""
FAISS vector-store manager.

Handles creation, persistence, document addition, and similarity
search against the FAISS index.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.rag.embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages a persistent FAISS vector store.

    Parameters:
        index_path: Directory where the FAISS index is saved / loaded.
        embedding_manager: Pre-initialised ``EmbeddingManager`` instance.
    """

    def __init__(
        self,
        index_path: str,
        embedding_manager: EmbeddingManager,
        distance_threshold: Optional[float] = None,
    ) -> None:
        self._index_path = Path(index_path)
        self._embedding_manager = embedding_manager
        self._distance_threshold = distance_threshold
        self._vectorstore: Optional[FAISS] = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialise(self) -> None:
        """Load an existing FAISS index from disk, or start empty."""
        async with self._lock:
            if self._index_path.exists() and (self._index_path / "index.faiss").exists():
                logger.info("Loading existing FAISS index from %s", self._index_path)
                self._vectorstore = await asyncio.to_thread(
                    FAISS.load_local,
                    str(self._index_path),
                    self._embedding_manager.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded successfully")
            else:
                logger.info("No existing index found – will create on first document upload")
                self._vectorstore = None

    async def _save(self) -> None:
        """Persist the current FAISS index to disk."""
        if self._vectorstore is None:
            return
        self._index_path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._vectorstore.save_local, str(self._index_path))
        logger.info("FAISS index saved to %s", self._index_path)

    # ── Public API ───────────────────────────────────────────────────────

    async def add_documents(self, documents: list[Document]) -> int:
        """Add document chunks to the vector store and persist.

        Returns:
            The number of chunks added.
        """
        if not documents:
            return 0

        async with self._lock:
            if self._vectorstore is None:
                logger.info("Creating new FAISS index with %d documents", len(documents))
                self._vectorstore = await asyncio.to_thread(
                    FAISS.from_documents,
                    documents,
                    self._embedding_manager.embeddings,
                )
            else:
                await asyncio.to_thread(self._vectorstore.add_documents, documents)
                logger.info("Added %d documents to existing index", len(documents))

            await self._save()
        return len(documents)

    async def search(self, query: str, k: int = 5) -> list[Document]:
        """Run similarity search and return the top-k matching documents."""
        if self._vectorstore is None:
            logger.warning("Vector store is empty – returning no results")
            return []

        if self._distance_threshold is not None:
            results_with_scores = await asyncio.to_thread(
                self._vectorstore.similarity_search_with_score, query, k=k
            )
            results = []
            for doc, score in results_with_scores:
                logger.debug("FAISS chunk distance score: %f", score)
                if score <= self._distance_threshold:
                    results.append(doc)
            logger.debug("Similarity search returned %d results for query (filtered from %d)", len(results), len(results_with_scores))
        else:
            results = await asyncio.to_thread(
                self._vectorstore.similarity_search, query, k=k
            )
            logger.debug("Similarity search returned %d results for query", len(results))
        return results

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Return a LangChain retriever backed by this FAISS store."""
        if self._vectorstore is None:
            return None
        kwargs = search_kwargs or {"k": 5}
        return self._vectorstore.as_retriever(search_kwargs=kwargs)

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None

    @property
    def index_path(self) -> str:
        return str(self._index_path)
