"""
FastAPI application entry point.

Creates the app, initialises all components during the lifespan,
and mounts the API router.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning import ReasoningAgent
from app.agents.research import ResearchAgent
from app.agents.router import RouterAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.tool_agent import ToolAgent
from app.api.routes import configure_routes, router
from app.config.settings import get_settings
from app.core.cache import QueryCache
from app.core.orchestrator import Orchestrator
from app.memory.conversation import ConversationMemory
from app.rag.document_processor import DocumentProcessor
from app.rag.embeddings import EmbeddingManager
from app.rag.vector_store import VectorStoreManager
from app.tools.code_executor import CodeExecutionTool
from app.tools.knowledge import KnowledgeTool
from app.tools.web_search import WebSearchTool

# ── Logging setup ────────────────────────────────────────────────────────────


def _configure_logging(level: str = "INFO") -> None:
    """Set up structured console logging."""
    log_format = (
        "%(asctime)s │ %(levelname)-8s │ %(name)-28s │ %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components on startup, clean up on shutdown."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("=" * 60)
    logger.info("  %s v%s", settings.app_title, settings.app_version)
    logger.info("=" * 60)

    # ── LLM ──────────────────────────────────────────────────────────
    logger.info("Initialising Groq LLM (model=%s)", settings.groq_model)
    llm = ChatGroq(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )

    # ── Embeddings & Vector Store ────────────────────────────────────
    logger.info("Loading embedding model: %s", settings.embedding_model)
    embedding_mgr = EmbeddingManager(model_name=settings.embedding_model)

    vector_store = VectorStoreManager(
        index_path=settings.faiss_index_path,
        embedding_manager=embedding_mgr,
    )
    await vector_store.initialise()

    # ── Document Processor ───────────────────────────────────────────
    doc_processor = DocumentProcessor(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # ── Tools ────────────────────────────────────────────────────────
    web_search = WebSearchTool(api_key=settings.tavily_api_key, max_results=3)
    knowledge_tool = KnowledgeTool(vector_store=vector_store, top_k=settings.rag_top_k)
    code_executor = CodeExecutionTool(timeout=settings.code_exec_timeout)

    # ── Memory & Cache ───────────────────────────────────────────────
    memory = ConversationMemory(max_size=settings.memory_size)
    cache = QueryCache(max_size=settings.cache_size, ttl=settings.cache_ttl)

    # ── Agents ───────────────────────────────────────────────────────
    planner = PlannerAgent(llm, settings)
    researcher = ResearchAgent(llm, settings, vector_store=vector_store, top_k=settings.rag_top_k)
    tool_agent = ToolAgent(
        llm, settings,
        web_search=web_search,
        knowledge_tool=knowledge_tool,
        code_executor=code_executor,
    )
    reasoner = ReasoningAgent(llm, settings)
    critic = CriticAgent(llm, settings, threshold=settings.critic_threshold)
    synthesizer = SynthesizerAgent(llm, settings)
    router = RouterAgent(llm, settings)

    # ── Orchestrator ─────────────────────────────────────────────────
    orchestrator = Orchestrator(
        planner=planner,
        researcher=researcher,
        tool_agent=tool_agent,
        reasoner=reasoner,
        critic=critic,
        synthesizer=synthesizer,
        router=router,
        memory=memory,
        cache=cache,
        settings=settings,
    )

    # ── Wire routes ──────────────────────────────────────────────────
    configure_routes(
        orchestrator=orchestrator,
        vector_store=vector_store,
        doc_processor=doc_processor,
        upload_dir=settings.upload_dir,
    )

    logger.info("All components initialised – system is READY")
    yield  # ← application runs here
    logger.info("Shutting down…")


# ── FastAPI app ──────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the router
app.include_router(router)


# ── Uvicorn runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
