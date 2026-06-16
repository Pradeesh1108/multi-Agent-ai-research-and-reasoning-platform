"""
DuckDuckGo web search tool.

Provides a simple interface to search the web and return the top
results as structured dictionaries.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import json
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Searches the web via DuckDuckGo and returns top results.

    Parameters:
        max_results: Number of results to return per search.
        timeout: Maximum seconds to wait for a search response.
    """

    def __init__(self, api_key: str, max_results: int = 3, timeout: int = 10) -> None:
        self._api_key = api_key
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
        """Synchronous Tavily search (runs in a thread)."""
        if not self._api_key:
            return [{"title": "API Error", "url": "", "snippet": "Tavily API key is missing."}]
            
        url = "https://api.tavily.com/search"
        data = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": self._max_results,
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
             raise RuntimeError(f"Tavily API error: {exc}")

        formatted: list[dict[str, str]] = []
        for r in result.get("results", []):
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
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
