"""
Knowledge retrieval tool backed by the FAISS vector store.

Provides a tool-agent-friendly interface to retrieve relevant
passages from the vector database.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class KnowledgeTool:
    """Retrieves relevant passages from the FAISS knowledge base.

    Parameters:
        vector_store: The ``VectorStoreManager`` instance backing retrieval.
        top_k: Default number of passages to retrieve.
    """

    def __init__(self, vector_store: VectorStoreManager, top_k: int = 5) -> None:
        self._vector_store = vector_store
        self._top_k = top_k

    # ── Public API ───────────────────────────────────────────────────────

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict[str, str]]:
        """Search the knowledge base and return matching passages.

        Returns:
            A list of dicts with keys ``content``, ``source``, and ``chunk_index``.
        """
        k = top_k or self._top_k
        logger.info("Knowledge retrieval: '%s' (k=%d)", query[:80], k)

        if not self._vector_store.is_ready:
            logger.warning("Knowledge base is empty – no documents uploaded yet")
            return []

        docs = await self._vector_store.search(query, k=k)

        results: list[dict[str, str]] = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "unknown"),
                "chunk_index": str(doc.metadata.get("chunk_index", -1)),
            })

        logger.info("Retrieved %d passages from knowledge base", len(results))
        return results

    def format_results(self, results: list[dict[str, str]]) -> str:
        """Format retrieved passages into a human-readable string."""
        if not results:
            return "No relevant passages found in the knowledge base."

        lines: list[str] = []
        for idx, r in enumerate(results, start=1):
            lines.append(
                f"[Passage {idx}] (Source: {r['source']}, Chunk: {r['chunk_index']})\n"
                f"{r['content']}"
            )
        return "\n\n".join(lines)

    @property
    def is_ready(self) -> bool:
        return self._vector_store.is_ready
