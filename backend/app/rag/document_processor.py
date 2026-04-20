"""
Document processor for ingesting PDFs and text files.

Loads documents, splits them into overlapping chunks, and returns
LangChain ``Document`` objects ready for embedding.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Supported file extensions → loader class mapping
_LOADERS: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": TextLoader,
}


class DocumentProcessor:
    """Ingests documents and splits them into retrieval-friendly chunks.

    Parameters:
        chunk_size: Target size of each text chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Public API ───────────────────────────────────────────────────────

    async def process_file(self, file_path: str) -> list[Document]:
        """Load and chunk a single file.

        Raises:
            ValueError: If the file type is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        loader_cls = _LOADERS.get(suffix)
        if loader_cls is None:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported types: {', '.join(_LOADERS.keys())}"
            )

        logger.info("Processing file: %s (type: %s)", path.name, suffix)

        # Load the raw documents (blocking I/O → thread pool)
        loader = loader_cls(str(path))
        raw_docs: list[Document] = await asyncio.to_thread(loader.load)
        logger.info("Loaded %d raw page(s) from %s", len(raw_docs), path.name)

        # Split into chunks
        chunks = self._splitter.split_documents(raw_docs)
        logger.info("Split into %d chunks (size=%d, overlap=%d)",
                     len(chunks), self._splitter._chunk_size, self._splitter._chunk_overlap)

        # Tag each chunk with source metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": path.name,
                "chunk_index": idx,
            })

        return chunks

    async def process_text(self, text: str, source_name: str = "direct_input") -> list[Document]:
        """Chunk raw text content directly (no file required)."""
        if not text.strip():
            return []

        doc = Document(page_content=text, metadata={"source": source_name})
        chunks = self._splitter.split_documents([doc])

        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "source_file": source_name,
                "chunk_index": idx,
            })

        logger.info("Processed raw text into %d chunks", len(chunks))
        return chunks
