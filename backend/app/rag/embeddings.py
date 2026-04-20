"""
HuggingFace embedding manager.

Uses the sentence-transformers library to run embeddings locally on
CPU (or GPU if available). Implements a singleton pattern so the
model is loaded only once per process.
"""

from __future__ import annotations

import logging
from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Singleton manager for the HuggingFace embedding model.

    Parameters:
        model_name: The HuggingFace model identifier
                    (default: ``sentence-transformers/all-MiniLM-L6-v2``).
    """

    _instance: EmbeddingManager | None = None
    _lock = Lock()

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingManager:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialised = False
                cls._instance = instance
            return cls._instance

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        if self._initialised:
            return
        logger.info("Loading embedding model: %s", model_name)
        self._model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._initialised = True
        logger.info("Embedding model loaded successfully")

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Return the underlying LangChain embeddings object."""
        return self._embeddings

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        return self._embeddings.embed_documents(texts)
