"""
Router Agent – evaluates queries to determine if the full pipeline is needed.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import RouterResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Router Agent in a multi-agent AI system. 
Your job is to decide if a user's query requires deep research, external tools, or multi-step reasoning.

Rules:
1. If the query is simple (e.g. conversational greetings, standard coding questions that you can answer immediately, or basic facts), route it as "direct" and provide the full answer in 'direct_answer'.
2. If the query is complex or specific to the user's context, route it as "complex".
   - CRITICAL: If the query mentions or implies uploaded documents, files, resumes, PDFs, or internal knowledge, you MUST set "needs_research" to true.
   - If the query requires current web search, external news, or sandbox code execution, set "needs_tools" to true.
3. Return ONLY valid JSON – no commentary outside the JSON block.

Output format (JSON):
{
  "route": "direct", 
  "direct_answer": "Your comprehensive answer here if direct...",
  "needs_research": false,
  "needs_tools": false
}
OR
{
  "route": "complex",
  "direct_answer": "",
  "needs_research": true,
  "needs_tools": false
}
"""

_USER_PROMPT = """\
## User Query
{query}

## Conversation Context
{context}

Decide the route and provide the JSON output.
"""

class RouterAgent(BaseAgent):
    """Determines if a query requires the full pipeline."""

    def __init__(self, llm, settings) -> None:
        super().__init__(llm, settings, name="RouterAgent")

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the router.

        Expected *input_data* keys:
            - ``query`` (str): The user's original question.
            - ``context`` (str, optional): Conversation history.

        Returns:
            A dict with key ``router_result`` containing a ``RouterResult`` model.
        """
        query: str = input_data["query"]
        context: str = input_data.get("context", "No prior context.")

        self._logger.info("Routing query: %s", query[:100])

        user_prompt = self._build_prompt(_USER_PROMPT, query=query, context=context)
        raw = await self._call_llm(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json_response(raw)

        if isinstance(parsed, dict):
            route = parsed.get("route", "complex").lower()
            if route not in ["direct", "complex"]:
                route = "complex"
            direct_answer = parsed.get("direct_answer", "")
            needs_research = bool(parsed.get("needs_research", True))
            needs_tools = bool(parsed.get("needs_tools", True))
        else:
            route = "complex"
            direct_answer = ""
            needs_research = True
            needs_tools = True

        router_result = RouterResult(
            route=route, 
            direct_answer=direct_answer,
            needs_research=needs_research,
            needs_tools=needs_tools,
        )
        self._logger.info("Router decision: %s", route)
        return {"router_result": router_result}
