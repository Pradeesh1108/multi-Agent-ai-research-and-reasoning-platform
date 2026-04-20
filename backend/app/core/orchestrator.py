"""
Central orchestrator that drives the multi-agent pipeline.

Responsibilities:
- Cache lookup before executing
- Sequential agent calls with timeout
- Retry loop when the Critic rejects an answer
- Memory storage after completion
- Full structured logging
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.agents.critic import CriticAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning import ReasoningAgent
from app.agents.research import ResearchAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.tool_agent import ToolAgent
from app.config.settings import Settings
from app.core.cache import QueryCache
from app.core.models import (
    CriticFeedback,
    PipelineState,
    Plan,
    QueryResponse,
    ReasoningOutput,
    ResearchResult,
    SynthesizedAnswer,
    ToolResult,
)
from app.memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the full agent pipeline from query to response.

    Parameters:
        planner: PlannerAgent instance.
        researcher: ResearchAgent instance.
        tool_agent: ToolAgent instance.
        reasoner: ReasoningAgent instance.
        critic: CriticAgent instance.
        synthesizer: SynthesizerAgent instance.
        memory: Shared ConversationMemory.
        cache: Shared QueryCache.
        settings: Application settings.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        researcher: ResearchAgent,
        tool_agent: ToolAgent,
        reasoner: ReasoningAgent,
        critic: CriticAgent,
        synthesizer: SynthesizerAgent,
        memory: ConversationMemory,
        cache: QueryCache,
        settings: Settings,
    ) -> None:
        self._planner = planner
        self._researcher = researcher
        self._tool_agent = tool_agent
        self._reasoner = reasoner
        self._critic = critic
        self._synthesizer = synthesizer
        self._memory = memory
        self._cache = cache
        self._settings = settings
        self._timeout = settings.agent_timeout

    # ── Main entry point ─────────────────────────────────────────────────

    async def process_query(self, query: str) -> QueryResponse:
        """Run the full pipeline for a user query.

        Returns a ``QueryResponse`` with all intermediate outputs.
        """
        start = time.time()
        logger.info("=" * 60)
        logger.info("PIPELINE START: %s", query[:120])
        logger.info("=" * 60)

        # ── 1. Cache check ───────────────────────────────────────────────
        cached = await self._cache.get(query)
        if cached is not None:
            logger.info("Cache HIT – returning stored response")
            cached.cached = True
            cached.processing_time_seconds = round(time.time() - start, 3)
            return cached

        # ── 2. Get conversation context ──────────────────────────────────
        context = await self._memory.get_context()

        # ── 3. Planner ──────────────────────────────────────────────────
        plan: Plan = await self._run_agent(
            "Planner",
            self._planner,
            {"query": query, "context": context},
            key="plan",
        )
        plan_text = "\n".join(
            f"{s.step_number}. {s.description}" for s in plan.steps
        )

        # ── 4. Research ─────────────────────────────────────────────────
        research: ResearchResult = await self._run_agent(
            "Research",
            self._researcher,
            {"query": query, "plan": plan},
            key="research",
        )
        research_text = research.summary
        if research.retrieved_passages:
            research_text += "\n\nPassages:\n" + "\n---\n".join(
                research.retrieved_passages[:5]
            )

        # ── 5. Tool Agent ───────────────────────────────────────────────
        tool_result: ToolResult = await self._run_agent(
            "Tool",
            self._tool_agent,
            {
                "query": query,
                "plan": plan,
                "research_context": research_text,
            },
            key="tool_result",
        )
        tool_text = tool_result.summary

        # ── 6. Reasoning → Critic loop ──────────────────────────────────
        reasoning: ReasoningOutput | None = None
        critic: CriticFeedback | None = None
        retries = 0
        critic_feedback_for_retry = "None – first pass."

        for attempt in range(1 + self._settings.max_retries):
            logger.info("Reasoning attempt %d / %d", attempt + 1, 1 + self._settings.max_retries)

            # Reasoning
            reasoning = await self._run_agent(
                "Reasoning",
                self._reasoner,
                {
                    "query": query,
                    "plan_text": plan_text,
                    "research_text": research_text,
                    "tool_text": tool_text,
                    "critic_feedback": critic_feedback_for_retry,
                },
                key="reasoning",
            )
            reasoning_steps = "\n".join(reasoning.steps)

            # Critic
            critic = await self._run_agent(
                "Critic",
                self._critic,
                {
                    "query": query,
                    "plan_text": plan_text,
                    "research_text": research_text,
                    "tool_text": tool_text,
                    "reasoning_steps": reasoning_steps,
                    "conclusion": reasoning.conclusion,
                },
                key="critic",
            )

            if critic.is_acceptable:
                logger.info("Critic APPROVED (score=%.1f)", critic.score)
                break
            else:
                retries += 1
                critic_feedback_for_retry = (
                    f"Previous score: {critic.score}/10\n"
                    f"Issues: {'; '.join(critic.issues)}\n"
                    f"Suggestions: {'; '.join(critic.suggestions)}"
                )
                logger.warning(
                    "Critic REJECTED (score=%.1f) – retrying (%d/%d)",
                    critic.score, retries, self._settings.max_retries,
                )

        # ── 7. Synthesizer ──────────────────────────────────────────────
        synthesized: SynthesizedAnswer = await self._run_agent(
            "Synthesizer",
            self._synthesizer,
            {
                "query": query,
                "plan_text": plan_text,
                "research_text": research_text,
                "tool_text": tool_text,
                "reasoning_steps": "\n".join(reasoning.steps) if reasoning else "",
                "conclusion": reasoning.conclusion if reasoning else "",
                "critic_score": str(critic.score) if critic else "N/A",
                "critic_feedback": critic.feedback_summary if critic else "",
                "critic_suggestions": "; ".join(critic.suggestions) if critic else "",
            },
            key="synthesized",
        )

        # ── 8. Build response ───────────────────────────────────────────
        elapsed = round(time.time() - start, 3)

        response = QueryResponse(
            query=query,
            plan=plan.summary + "\n" + plan_text,
            research_context=research_text[:2000],
            tool_results=tool_text[:2000],
            reasoning=reasoning.conclusion if reasoning else "",
            critic_feedback=(
                f"Score: {critic.score}/10 – {critic.feedback_summary}"
                if critic else ""
            ),
            answer=synthesized.answer,
            retries_used=retries,
            cached=False,
            processing_time_seconds=elapsed,
        )

        # ── 9. Store in cache and memory ────────────────────────────────
        await self._cache.set(query, response)
        await self._memory.add_interaction(query, synthesized.answer)

        logger.info("PIPELINE COMPLETE in %.2fs (retries=%d)", elapsed, retries)
        return response

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _run_agent(
        self,
        label: str,
        agent: Any,
        input_data: dict[str, Any],
        key: str,
    ) -> Any:
        """Execute a single agent with timeout and error handling."""
        logger.info("▶ Running %s Agent...", label)
        try:
            result = await asyncio.wait_for(
                agent.execute(input_data),
                timeout=self._timeout,
            )
            output = result.get(key)
            logger.info("✓ %s Agent completed", label)
            return output
        except asyncio.TimeoutError:
            logger.error("✗ %s Agent timed out after %ds", label, self._timeout)
            raise RuntimeError(f"{label} Agent timed out after {self._timeout}s")
        except Exception as exc:
            logger.error("✗ %s Agent failed: %s", label, exc, exc_info=True)
            raise RuntimeError(f"{label} Agent failed: {exc}") from exc
