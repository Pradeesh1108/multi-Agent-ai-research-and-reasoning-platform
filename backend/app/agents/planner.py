"""
Planner Agent – breaks a user query into a structured execution plan.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import Plan, PlanStep

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Planner Agent in a multi-agent AI system. Your sole responsibility
is to decompose a user's query into a structured, actionable plan.

Rules:
1. Produce between 2 and 8 steps.
2. Each step must be clear, specific, and self-contained.
3. Order the steps logically – dependencies first.
4. For each step, optionally suggest which downstream agent should handle it
   (research, tool, reasoning, or synthesis).
5. Return ONLY valid JSON – no commentary outside the JSON block.

Output format (JSON):
{
  "summary": "<one-sentence summary of the plan>",
  "steps": [
    {"step_number": 1, "description": "...", "agent_hint": "research"},
    {"step_number": 2, "description": "...", "agent_hint": "tool"},
    ...
  ]
}
"""

_USER_PROMPT = """\
## User Query
{query}

## Conversation Context
{context}

Generate the execution plan as described.
"""


class PlannerAgent(BaseAgent):
    """Decomposes a user query into a structured list of steps."""

    def __init__(self, llm, settings) -> None:
        super().__init__(llm, settings, name="PlannerAgent")

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the planner.

        Expected *input_data* keys:
            - ``query`` (str): The user's original question.
            - ``context`` (str, optional): Conversation history.

        Returns:
            A dict with key ``plan`` containing a ``Plan`` model.
        """
        query: str = input_data["query"]
        context: str = input_data.get("context", "No prior context.")

        self._logger.info("Planning for query: %s", query[:100])

        user_prompt = self._build_prompt(_USER_PROMPT, query=query, context=context)
        raw = await self._call_llm(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json_response(raw)

        # Build typed model
        if isinstance(parsed, dict):
            steps = [
                PlanStep(
                    step_number=s.get("step_number", idx + 1),
                    description=s.get("description", ""),
                    agent_hint=s.get("agent_hint"),
                )
                for idx, s in enumerate(parsed.get("steps", []))
            ]
            summary = parsed.get("summary", "Plan generated.")
        else:
            # Fallback – treat entire text as a single step
            steps = [PlanStep(step_number=1, description=str(parsed))]
            summary = "Single-step plan."

        plan = Plan(original_query=query, steps=steps, summary=summary)
        self._logger.info("Plan created with %d steps", len(plan.steps))
        return {"plan": plan}
