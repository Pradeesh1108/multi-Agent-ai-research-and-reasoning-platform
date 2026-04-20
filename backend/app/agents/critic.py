"""
Critic Agent – evaluates the quality of reasoning and detects issues.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import CriticFeedback

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Critic Agent in a multi-agent AI system. Your job is to
rigorously evaluate the quality of the reasoning and answer produced
by the Reasoning Agent.

Evaluate the following dimensions (each on a 1-10 scale):
1. **Accuracy** – Are claims factually correct and well-supported?
2. **Completeness** – Does the answer address all parts of the query?
3. **Coherence** – Is the reasoning logical and well-structured?
4. **Hallucination Risk** – Is there speculation presented as fact?
5. **Relevance** – Does the answer stay focused on the query?

Rules:
- Be strict and objective.
- Identify specific issues with exact quotes when possible.
- Provide actionable improvement suggestions.
- Compute an overall score (average of all dimensions).
- Return ONLY valid JSON.

Output format:
{
  "score": 7.5,
  "dimensions": {
    "accuracy": 8,
    "completeness": 7,
    "coherence": 8,
    "hallucination_risk": 7,
    "relevance": 8
  },
  "issues": [
    "Issue 1 description",
    "Issue 2 description"
  ],
  "suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ],
  "feedback_summary": "Overall assessment in 2-3 sentences."
}
"""

_USER_PROMPT = """\
## Original Query
{query}

## Plan
{plan}

## Research Context
{research}

## Tool Outputs
{tool_output}

## Reasoning Output
### Steps:
{reasoning_steps}

### Conclusion:
{conclusion}

Evaluate the quality of this reasoning and answer.
"""


class CriticAgent(BaseAgent):
    """Evaluates reasoning output for quality, accuracy, and completeness."""

    def __init__(self, llm, settings, threshold: float = 7.0) -> None:
        super().__init__(llm, settings, name="CriticAgent")
        self._threshold = threshold

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the critic evaluation.

        Expected *input_data* keys:
            - ``query`` (str)
            - ``plan_text`` (str)
            - ``research_text`` (str)
            - ``tool_text`` (str)
            - ``reasoning_steps`` (str)
            - ``conclusion`` (str)

        Returns:
            A dict with key ``critic`` containing a ``CriticFeedback``.
        """
        query: str = input_data["query"]

        self._logger.info("Evaluating answer quality for query: %s", query[:100])

        user_prompt = self._build_prompt(
            _USER_PROMPT,
            query=query,
            plan=input_data.get("plan_text", ""),
            research=input_data.get("research_text", "")[:2000],
            tool_output=input_data.get("tool_text", "")[:2000],
            reasoning_steps=input_data.get("reasoning_steps", ""),
            conclusion=input_data.get("conclusion", ""),
        )

        raw = await self._call_llm(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json_response(raw)

        if isinstance(parsed, dict):
            score = float(parsed.get("score", 5.0))
            score = max(0.0, min(10.0, score))
            issues = parsed.get("issues", [])
            suggestions = parsed.get("suggestions", [])
            feedback_summary = parsed.get("feedback_summary", "")
        else:
            score = 5.0
            issues = ["Could not parse critic evaluation"]
            suggestions = ["Retry with clearer reasoning"]
            feedback_summary = str(parsed)

        is_acceptable = score >= self._threshold

        critic = CriticFeedback(
            score=score,
            is_acceptable=is_acceptable,
            issues=issues if isinstance(issues, list) else [str(issues)],
            suggestions=suggestions if isinstance(suggestions, list) else [str(suggestions)],
            feedback_summary=feedback_summary,
        )

        self._logger.info(
            "Critic evaluation: score=%.1f, acceptable=%s, issues=%d",
            critic.score, critic.is_acceptable, len(critic.issues),
        )
        return {"critic": critic}
