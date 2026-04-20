"""
Synthesizer Agent – produces the final polished answer.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import SynthesizedAnswer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Synthesizer Agent – the final stage of a multi-agent AI pipeline.
Your job is to produce a clear, comprehensive, and well-structured final answer.

You receive the full context from all previous agents:
- The execution plan
- Research findings
- Tool outputs
- Detailed reasoning
- Critic feedback and scores

Rules:
1. Write the answer for a human reader – clear, well-organised, no jargon.
2. Incorporate improvements suggested by the Critic.
3. Highlight key points and takeaways.
4. Cite sources when available.
5. If there are uncertainties, state them honestly.
6. Return ONLY valid JSON.

Output format:
{
  "answer": "<complete, polished answer>",
  "key_points": [
    "Key point 1",
    "Key point 2"
  ],
  "sources_used": [
    "Source 1",
    "Source 2"
  ]
}
"""

_USER_PROMPT = """\
## Original Query
{query}

## Execution Plan
{plan}

## Research Findings
{research}

## Tool Outputs
{tool_output}

## Reasoning
### Steps:
{reasoning_steps}

### Conclusion:
{conclusion}

## Critic Feedback
Score: {critic_score}/10
Feedback: {critic_feedback}
Suggestions: {critic_suggestions}

Produce the final, polished answer.
"""


class SynthesizerAgent(BaseAgent):
    """Combines all agent outputs into a final, high-quality answer."""

    def __init__(self, llm, settings) -> None:
        super().__init__(llm, settings, name="SynthesizerAgent")

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run synthesis.

        Expected *input_data* keys:
            - ``query``, ``plan_text``, ``research_text``, ``tool_text``
            - ``reasoning_steps``, ``conclusion``
            - ``critic_score``, ``critic_feedback``, ``critic_suggestions``

        Returns:
            A dict with key ``synthesized`` containing a ``SynthesizedAnswer``.
        """
        query: str = input_data["query"]

        self._logger.info("Synthesizing final answer for: %s", query[:100])

        user_prompt = self._build_prompt(
            _USER_PROMPT,
            query=query,
            plan=input_data.get("plan_text", ""),
            research=input_data.get("research_text", "")[:3000],
            tool_output=input_data.get("tool_text", "")[:3000],
            reasoning_steps=input_data.get("reasoning_steps", ""),
            conclusion=input_data.get("conclusion", ""),
            critic_score=input_data.get("critic_score", "N/A"),
            critic_feedback=input_data.get("critic_feedback", ""),
            critic_suggestions=input_data.get("critic_suggestions", ""),
        )

        raw = await self._call_llm(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json_response(raw)

        if isinstance(parsed, dict):
            answer = parsed.get("answer", "")
            key_points = parsed.get("key_points", [])
            sources = parsed.get("sources_used", [])
        else:
            answer = str(parsed)
            key_points = []
            sources = []

        synthesized = SynthesizedAnswer(
            answer=answer,
            key_points=key_points if isinstance(key_points, list) else [str(key_points)],
            sources_used=sources if isinstance(sources, list) else [str(sources)],
        )
        self._logger.info("Synthesis complete (%d chars, %d key points)",
                          len(synthesized.answer), len(synthesized.key_points))
        return {"synthesized": synthesized}
