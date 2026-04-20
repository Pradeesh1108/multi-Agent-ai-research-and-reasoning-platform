"""
Reasoning Agent – performs structured chain-of-thought analysis.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import ReasoningOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Reasoning Agent in a multi-agent AI system. Your job is to
perform deep, step-by-step analytical reasoning to answer the user's query.

You receive:
- The original query
- A structured plan
- Research context (retrieved passages)
- Tool outputs (web search, code execution, etc.)
- Any previous critic feedback (if this is a retry)

Rules:
1. Reason through the problem step by step.
2. Cross-reference information from multiple sources.
3. Identify and resolve contradictions.
4. Be explicit about assumptions and uncertainties.
5. Provide a clear conclusion.
6. Return ONLY valid JSON.

Output format:
{
  "steps": [
    "Step 1: <reasoning step>",
    "Step 2: <reasoning step>",
    ...
  ],
  "conclusion": "<final reasoned conclusion>",
  "confidence": 0.85
}
"""

_USER_PROMPT = """\
## User Query
{query}

## Execution Plan
{plan}

## Research Context
{research}

## Tool Outputs
{tool_output}

## Previous Critic Feedback (if any)
{critic_feedback}

Perform your step-by-step reasoning and provide a well-structured conclusion.
"""


class ReasoningAgent(BaseAgent):
    """Combines all gathered information into structured reasoning."""

    def __init__(self, llm, settings) -> None:
        super().__init__(llm, settings, name="ReasoningAgent")

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run chain-of-thought reasoning.

        Expected *input_data* keys:
            - ``query`` (str)
            - ``plan_text`` (str)
            - ``research_text`` (str)
            - ``tool_text`` (str)
            - ``critic_feedback`` (str, optional) – for retries

        Returns:
            A dict with key ``reasoning`` containing a ``ReasoningOutput``.
        """
        query: str = input_data["query"]
        plan_text: str = input_data.get("plan_text", "")
        research_text: str = input_data.get("research_text", "")
        tool_text: str = input_data.get("tool_text", "")
        critic_feedback: str = input_data.get("critic_feedback", "None – first pass.")

        self._logger.info("Running reasoning for query: %s", query[:100])

        user_prompt = self._build_prompt(
            _USER_PROMPT,
            query=query,
            plan=plan_text,
            research=research_text[:3000],
            tool_output=tool_text[:3000],
            critic_feedback=critic_feedback,
        )

        raw = await self._call_llm(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json_response(raw)

        if isinstance(parsed, dict):
            steps = parsed.get("steps", [])
            conclusion = parsed.get("conclusion", "")
            confidence = parsed.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence = max(0.0, min(1.0, float(confidence)))
            else:
                confidence = None
        else:
            steps = [str(parsed)]
            conclusion = str(parsed)
            confidence = None

        reasoning = ReasoningOutput(
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
        )
        self._logger.info(
            "Reasoning complete: %d steps, confidence=%s",
            len(reasoning.steps),
            reasoning.confidence,
        )
        return {"reasoning": reasoning}
