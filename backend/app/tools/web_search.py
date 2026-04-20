"""
DuckDuckGo web search tool.

Provides a simple interface to search the web and return the top
results as structured dictionaries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Searches the web via DuckDuckGo and returns top results.

    Parameters:
        max_results: Number of results to return per search.
        timeout: Maximum seconds to wait for a search response.
    """

    def __init__(self, max_results: int = 3, timeout: int = 10) -> None:
        self._max_results = max_results
        self._timeout = timeout

    # ── Public API ───────────────────────────────────────────────────────

    async def search(self, query: str) -> list[dict[str, str]]:
        """Execute a web search and return structured results.

        Returns:
            A list of dicts with keys ``title``, ``url``, and ``snippet``.
        """
        logger.info("Web search: %s (max_results=%d)", query, self._max_results)
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(self._sync_search, query),
                timeout=self._timeout,
            )
            logger.info("Web search returned %d results", len(results))
            return results
        except asyncio.TimeoutError:
            logger.warning("Web search timed out after %ds", self._timeout)
            return [{"title": "Search Timeout", "url": "", "snippet": "The web search timed out."}]
        except Exception as exc:
            logger.error("Web search failed: %s", exc, exc_info=True)
            return [{"title": "Search Error", "url": "", "snippet": str(exc)}]

    # ── Internal ─────────────────────────────────────────────────────────

    def _sync_search(self, query: str) -> list[dict[str, str]]:
        """Synchronous DuckDuckGo search (runs in a thread)."""
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=self._max_results)

        formatted: list[dict[str, str]] = []
        for r in raw_results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "snippet": r.get("body", r.get("snippet", "")),
            })
        return formatted

    def format_results(self, results: list[dict[str, str]]) -> str:
        """Format search results into a human-readable string."""
        if not results:
            return "No web search results found."

        lines: list[str] = []
        for idx, r in enumerate(results, start=1):
            lines.append(
                f"{idx}. **{r['title']}**\n"
                f"   URL: {r['url']}\n"
                f"   {r['snippet']}"
            )
        return "\n\n".join(lines)
