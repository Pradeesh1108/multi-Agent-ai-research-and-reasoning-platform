"""
Pydantic models for API request/response schemas and inter-agent data transfer.

Every piece of data flowing through the pipeline is strongly typed, which
makes debugging, logging, and serialisation straightforward.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── API Schemas ──────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Incoming user query."""
    query: str = Field(..., min_length=1, max_length=5000, description="The user query to process")


class QueryResponse(BaseModel):
    """Final structured response returned by the /query endpoint."""
    query: str
    plan: str
    research_context: str
    tool_results: str
    reasoning: str
    critic_feedback: str
    answer: str
    retries_used: int = 0
    cached: bool = False
    processing_time_seconds: float = 0.0


class UploadResponse(BaseModel):
    """Response after uploading and ingesting a document."""
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    timestamp: str
    components: dict[str, str]


# ── Inter-Agent Data Models ──────────────────────────────────────────────────


class RouterResult(BaseModel):
    """Output from the Router Agent determining the pipeline path."""
    route: str = Field(..., description="'direct' or 'complex'")
    direct_answer: str = ""
    needs_research: bool = True
    needs_tools: bool = True


class PlanStep(BaseModel):
    """A single step in the plan produced by the Planner Agent."""
    step_number: int
    description: str
    agent_hint: Optional[str] = None  # which agent this step targets


class Plan(BaseModel):
    """Structured plan output from the Planner Agent."""
    original_query: str
    steps: list[PlanStep]
    summary: str


class ResearchResult(BaseModel):
    """Aggregated research output from the Research Agent."""
    query: str
    retrieved_passages: list[str]
    source_count: int
    summary: str


class ToolResultItem(BaseModel):
    """Result from a single tool invocation."""
    tool_name: str
    input_data: str
    output_data: str
    success: bool
    error: Optional[str] = None


class ToolResult(BaseModel):
    """Aggregated tool output from the Tool Agent."""
    tools_used: list[str]
    results: list[ToolResultItem]
    summary: str


class ReasoningOutput(BaseModel):
    """Step-by-step reasoning from the Reasoning Agent."""
    steps: list[str]
    conclusion: str
    confidence: Optional[float] = None


class CriticFeedback(BaseModel):
    """Evaluation from the Critic Agent."""
    score: float = Field(..., ge=0, le=10)
    is_acceptable: bool
    issues: list[str]
    suggestions: list[str]
    feedback_summary: str


class SynthesizedAnswer(BaseModel):
    """Final polished answer from the Synthesizer Agent."""
    answer: str
    key_points: list[str]
    sources_used: list[str]


# ── Pipeline State ───────────────────────────────────────────────────────────


class PipelineState(BaseModel):
    """Full state of the agent pipeline at any point in execution."""
    query: str
    conversation_context: str = ""
    router_result: Optional[RouterResult] = None
    plan: Optional[Plan] = None
    research: Optional[ResearchResult] = None
    tool_output: Optional[ToolResult] = None
    reasoning: Optional[ReasoningOutput] = None
    critic: Optional[CriticFeedback] = None
    final_answer: Optional[SynthesizedAnswer] = None
    retries: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    errors: list[str] = Field(default_factory=list)
