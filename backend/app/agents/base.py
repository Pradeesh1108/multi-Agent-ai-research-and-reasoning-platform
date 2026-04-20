"""
Abstract base class for all agents.

Provides shared infrastructure: LLM access, structured prompt building,
logging, error handling, and a standardised ``execute()`` interface.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import Settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class that every specialised agent inherits from.

    Parameters:
        llm: A shared ``ChatGroq`` instance.
        settings: Application settings.
        name: Human-readable agent name (used in logs).
    """

    def __init__(self, llm: ChatGroq, settings: Settings, name: str = "BaseAgent") -> None:
        self._llm = llm
        self._settings = settings
        self._name = name
        self._logger = logging.getLogger(f"agent.{name}")

    # ── Abstract contract ────────────────────────────────────────────────

    @abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the agent on *input_data* and return structured output."""
        ...

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send a system + user message pair to the LLM and return the text."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        self._logger.debug("Calling LLM (system=%d chars, user=%d chars)",
                           len(system_prompt), len(user_prompt))
        response = await self._llm.ainvoke(messages)
        text = response.content
        self._logger.debug("LLM response: %d chars", len(text))
        return text

    @staticmethod
    def _parse_json_response(text: str) -> Any:
        """Best-effort extraction of a JSON object / array from LLM text.

        Handles common LLM quirks: markdown fences, trailing commas,
        escaped quotes, and text outside the JSON block.
        """
        # Try raw parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding the first { … } or [ … ]
        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                for attempt_text in [
                    candidate,
                    re.sub(r",\s*([}\]])", r"\1", candidate),       # trailing commas
                    candidate.replace('\\"', "'"),                    # over-escaped quotes
                    re.sub(r'\\{2,}"', '"', candidate),              # multiple backslash escapes
                ]:
                    try:
                        return json.loads(attempt_text)
                    except json.JSONDecodeError:
                        continue

        # Give up – return raw text
        logger.warning("Could not parse JSON from LLM response; returning raw text")
        return text

    @staticmethod
    def _build_prompt(template: str, **kwargs: Any) -> str:
        """Format a prompt template with keyword arguments."""
        try:
            return template.format(**kwargs)
        except KeyError as exc:
            logger.error("Missing prompt variable: %s", exc)
            raise

    @property
    def name(self) -> str:
        return self._name
