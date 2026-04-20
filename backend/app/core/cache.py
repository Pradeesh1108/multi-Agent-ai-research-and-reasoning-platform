"""
In-memory LRU cache with TTL expiry for query responses.

Provides fast lookups for repeated queries while automatically evicting
stale entries after the configured time-to-live.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryCache:
    """Thread-safe async LRU cache with TTL support.

    Attributes:
        max_size: Maximum number of entries in the cache.
        ttl: Time-to-live in seconds for each entry.
    """

    def __init__(self, max_size: int = 100, ttl: int = 300) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    # ── Public API ───────────────────────────────────────────────────────

    @staticmethod
    def _make_key(query: str) -> str:
        """Create a deterministic hash key from a query string."""
        normalised = query.strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    async def get(self, query: str) -> Optional[Any]:
        """Retrieve a cached response, or *None* if missing / expired."""
        key = self._make_key(query)
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            timestamp, value = self._cache[key]
            if time.time() - timestamp > self._ttl:
                # Expired – evict
                del self._cache[key]
                self._misses += 1
                logger.debug("Cache entry expired for key %s", key[:12])
                return None

            # Move to end (most-recently-used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit for key %s", key[:12])
            return value

    async def set(self, query: str, value: Any) -> None:
        """Store a response in the cache, evicting the oldest entry if full."""
        key = self._make_key(query)
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), value)

            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Evicted cache entry %s", evicted_key[:12])

    async def invalidate(self, query: str) -> bool:
        """Remove a specific entry from the cache.  Returns True if found."""
        key = self._make_key(query)
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear the entire cache.  Returns the number of entries removed."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared – removed %d entries", count)
            return count

    @property
    def stats(self) -> dict[str, int]:
        """Return cache hit / miss statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
        }
