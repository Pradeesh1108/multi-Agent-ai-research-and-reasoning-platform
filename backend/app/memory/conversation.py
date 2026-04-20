"""
Sliding-window conversation memory.

Stores the last *N* interactions to provide conversational context
to agents without unbounded growth.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Thread-safe, async-compatible sliding-window memory.

    Parameters:
        max_size: Maximum number of interactions to retain.
    """

    def __init__(self, max_size: int = 10) -> None:
        self._max_size = max_size
        self._history: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

    # ── Public API ───────────────────────────────────────────────────────

    async def add_interaction(self, query: str, response: str) -> None:
        """Record a new query / response pair."""
        async with self._lock:
            entry = {
                "query": query,
                "response": response,
                "timestamp": datetime.utcnow().isoformat(),
            }
            self._history.append(entry)
            logger.debug(
                "Memory: stored interaction (%d / %d)",
                len(self._history),
                self._max_size,
            )

    async def get_context(self) -> str:
        """Return a formatted string of the conversation history.

        Suitable for injecting directly into agent prompts.
        """
        async with self._lock:
            if not self._history:
                return "No previous conversation history."

            lines: list[str] = []
            for idx, entry in enumerate(self._history, start=1):
                lines.append(
                    f"[Turn {idx}]\n"
                    f"User: {entry['query']}\n"
                    f"Assistant: {entry['response']}\n"
                )
            return "\n".join(lines)

    async def get_recent(self, n: int = 3) -> list[dict[str, Any]]:
        """Return the *n* most recent interactions as dicts."""
        async with self._lock:
            items = list(self._history)
            return items[-n:]

    async def clear(self) -> int:
        """Clear the entire history.  Returns the number of entries removed."""
        async with self._lock:
            count = len(self._history)
            self._history.clear()
            logger.info("Memory cleared – removed %d entries", count)
            return count

    @property
    def size(self) -> int:
        return len(self._history)

    @property
    def max_size(self) -> int:
        return self._max_size
