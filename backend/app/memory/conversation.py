"""
Redis-backed session-aware conversation memory.

Stores the last *N* interactions per session in Redis to provide
conversational context to agents across multiple API servers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Async Redis-backed conversation memory.

    Parameters:
        redis_client: Initialized redis.asyncio.Redis connection.
        max_size: Maximum number of interactions to retain per session.
        ttl: Expiration time in seconds for a session's history.
    """

    def __init__(self, redis_client: Redis, max_size: int = 10, ttl: int = 86400) -> None:
        self._redis = redis_client
        self._max_size = max_size
        self._ttl = ttl

    # ── Public API ───────────────────────────────────────────────────────

    def _get_key(self, session_id: str) -> str:
        return f"chat_history:{session_id}"

    async def add_interaction(self, session_id: str, query: str, response: str) -> None:
        """Record a new query / response pair for a specific session."""
        key = self._get_key(session_id)
        entry = {
            "query": query,
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # We use a pipeline or transaction to ensure atomicity
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(entry))
            pipe.ltrim(key, -self._max_size, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()
            
        logger.debug("Memory: stored interaction for session '%s'", session_id)

    async def get_context(self, session_id: str) -> str:
        """Return a formatted string of the conversation history."""
        key = self._get_key(session_id)
        items = await self._redis.lrange(key, 0, -1)
        
        if not items:
            return "No previous conversation history."

        lines: list[str] = []
        for idx, item in enumerate(items, start=1):
            entry = json.loads(item)
            lines.append(
                f"[Turn {idx}]\n"
                f"User: {entry['query']}\n"
                f"Assistant: {entry['response']}\n"
            )
        return "\n".join(lines)

    async def get_recent(self, session_id: str, n: int = 3) -> list[dict[str, Any]]:
        """Return the *n* most recent interactions as dicts."""
        key = self._get_key(session_id)
        items = await self._redis.lrange(key, -n, -1)
        return [json.loads(item) for item in items]

    async def clear(self, session_id: str) -> int:
        """Clear the history for a specific session."""
        key = self._get_key(session_id)
        count = await self._redis.llen(key)
        await self._redis.delete(key)
        logger.info("Memory cleared for session '%s' – removed %d entries", session_id, count)
        return count

    @property
    def max_size(self) -> int:
        return self._max_size
