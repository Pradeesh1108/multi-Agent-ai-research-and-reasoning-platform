"""
Application settings loaded from environment variables.

Uses pydantic-settings to provide type-safe configuration with
automatic .env file loading and sensible defaults for all values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the multi-agent platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── FAISS / RAG ──────────────────────────────────────────────────────
    faiss_index_path: str = str(Path(__file__).resolve().parents[2] / "data" / "faiss_index")
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5

    # ── Uploads ──────────────────────────────────────────────────────────
    upload_dir: str = str(Path(__file__).resolve().parents[2] / "uploads")

    # ── Memory ───────────────────────────────────────────────────────────
    memory_size: int = 10

    # ── Orchestrator ─────────────────────────────────────────────────────
    max_retries: int = 2
    critic_threshold: float = 7.0
    agent_timeout: int = 60  # seconds per agent call

    # ── Cache ────────────────────────────────────────────────────────────
    cache_size: int = 100
    cache_ttl: int = 300  # seconds

    # ── Code Execution ───────────────────────────────────────────────────
    code_exec_timeout: int = 5  # seconds

    # ── Server ───────────────────────────────────────────────────────────
    app_title: str = "Multi-Agent AI Research Platform"
    app_version: str = "1.0.0"
    app_description: str = (
        "A production-ready multi-agent AI system for research, reasoning, "
        "and knowledge retrieval powered by Groq LLM."
    )
    cors_origins: list[str] = ["*"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
