"""
Research Agent – retrieves relevant knowledge from the FAISS vector store.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import Plan, ResearchResult
from app.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """Queries the vector knowledge base using plan steps as search terms.

    Parameters:
        vector_store: The FAISS ``VectorStoreManager`` instance.
        top_k: Number of passages to retrieve per search.
    """

    def __init__(self, llm, settings, vector_store: VectorStoreManager, top_k: int = 5) -> None:
        super().__init__(llm, settings, name="ResearchAgent")
        self._vector_store = vector_store
        self._top_k = top_k

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run research retrieval.

        Expected *input_data* keys:
            - ``query`` (str)
            - ``plan`` (Plan)

        Returns:
            A dict with key ``research`` containing a ``ResearchResult``.
        """
        query: str = input_data["query"]
        plan: Plan = input_data["plan"]

        self._logger.info("Researching %d plan steps", len(plan.steps))

        if not self._vector_store.is_ready:
            self._logger.warning("Vector store not ready – returning empty research")
            return {
                "research": ResearchResult(
                    query=query,
                    retrieved_passages=["No knowledge base available. Please upload documents first."],
                    source_count=0,
                    summary="Knowledge base is empty.",
                )
            }

        # Build search queries from plan steps + the original query
        search_queries = [query] + [step.description for step in plan.steps]

        all_passages: list[str] = []
        seen_contents: set[str] = set()

        for sq in search_queries:
            docs = await self._vector_store.search(sq, k=self._top_k)
            for doc in docs:
                content_key = doc.page_content.strip()[:200]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    source = doc.metadata.get("source_file", "unknown")
                    all_passages.append(
                        f"[Source: {source}] {doc.page_content.strip()}"
                    )

        # Use the LLM to summarize retrieved passages
        if all_passages:
            summary_prompt = (
                "Summarize the following retrieved passages in relation to this query:\n"
                f"Query: {query}\n\n"
                "Passages:\n" + "\n---\n".join(all_passages[:10])
            )
            summary = await self._call_llm(
                "You are a research summariser. Be concise and accurate.",
                summary_prompt,
            )
        else:
            summary = "No relevant passages found in the knowledge base."

        research = ResearchResult(
            query=query,
            retrieved_passages=all_passages[:15],  # cap at 15 unique passages
            source_count=len(all_passages),
            summary=summary,
        )
        self._logger.info("Research complete: %d unique passages retrieved", len(all_passages))
        return {"research": research}
