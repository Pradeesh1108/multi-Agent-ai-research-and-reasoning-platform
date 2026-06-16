"""
Tool Agent – dynamically selects and invokes tools based on the query.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import BaseAgent
from app.core.models import Plan, ToolResult, ToolResultItem
from app.tools.code_executor import CodeExecutionTool
from app.tools.knowledge import KnowledgeTool
from app.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the Tool Selection Agent. Given a user query and an execution plan,
decide which tools to invoke and what input to give each tool.

Available tools:
1. web_search – Search the internet for current information. Input: a search query string.
2. knowledge_retrieval – Search the internal knowledge base (uploaded documents). Input: a search query string.
3. code_execution – Execute a Python code snippet for computation or analysis. Input: a Python code string.

Rules:
- Select 1–3 tools that are most relevant.
- ONLY use web_search if the query specifically requires current events, external news, or facts you don't know. Do NOT use it if the answer is in the documents or can be computed.
- If the query relates to uploaded documents, use knowledge_retrieval.
- If the query requires computation, use code_execution.
- Return ONLY valid JSON.

Output format:
{
  "tools": [
    {"name": "web_search", "input": "search query here"},
    {"name": "code_execution", "input": "print(2 + 2)"}
  ]
}
"""

_USER_PROMPT = """\
## User Query
{query}

## Execution Plan
{plan}

Decide which tools to use and what inputs to provide.
"""


class ToolAgent(BaseAgent):
    """Dynamically selects and invokes tools to gather additional information.

    Parameters:
        web_search: ``WebSearchTool`` instance.
        knowledge_tool: ``KnowledgeTool`` instance.
        code_executor: ``CodeExecutionTool`` instance.
    """

    def __init__(
        self,
        llm,
        settings,
        web_search: WebSearchTool,
        knowledge_tool: KnowledgeTool,
        code_executor: CodeExecutionTool,
    ) -> None:
        super().__init__(llm, settings, name="ToolAgent")
        self._tools = {
            "web_search": web_search,
            "knowledge_retrieval": knowledge_tool,
            "code_execution": code_executor,
        }

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run tool selection and execution.

        Expected *input_data* keys:
            - ``query`` (str)
            - ``plan`` (Plan)
            - ``research_context`` (str)

        Returns:
            A dict with key ``tool_result`` containing a ``ToolResult``.
        """
        query: str = input_data["query"]
        plan: Plan = input_data["plan"]

        plan_text = "\n".join(
            f"{s.step_number}. {s.description}" for s in plan.steps
        )

        user_prompt = self._build_prompt(
            _USER_PROMPT,
            query=query,
            plan=plan_text,
        )

        raw = await self._call_llm(_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json_response(raw)

        tool_calls: list[dict] = []
        if isinstance(parsed, dict):
            tool_calls = parsed.get("tools", [])
        elif isinstance(parsed, list):
            tool_calls = parsed

        # Execute selected tools
        results: list[ToolResultItem] = []
        tools_used: list[str] = []

        for tc in tool_calls:
            tool_name = tc.get("name", "").strip()
            tool_input = tc.get("input", "").strip()

            if tool_name not in self._tools:
                self._logger.warning("Unknown tool requested: %s", tool_name)
                results.append(ToolResultItem(
                    tool_name=tool_name,
                    input_data=tool_input,
                    output_data="",
                    success=False,
                    error=f"Unknown tool: {tool_name}",
                ))
                continue

            self._logger.info("Invoking tool: %s", tool_name)
            tools_used.append(tool_name)

            try:
                output = await self._invoke_tool(tool_name, tool_input)
                results.append(ToolResultItem(
                    tool_name=tool_name,
                    input_data=tool_input,
                    output_data=str(output),
                    success=True,
                ))
            except Exception as exc:
                self._logger.error("Tool %s failed: %s", tool_name, exc)
                results.append(ToolResultItem(
                    tool_name=tool_name,
                    input_data=tool_input,
                    output_data="",
                    success=False,
                    error=str(exc),
                ))

        # Build summary
        summary_parts = []
        for r in results:
            status = "✓" if r.success else "✗"
            summary_parts.append(f"[{status}] {r.tool_name}: {r.output_data[:200]}")
        summary = "\n".join(summary_parts) if summary_parts else "No tools were invoked."

        tool_result = ToolResult(
            tools_used=tools_used,
            results=results,
            summary=summary,
        )
        self._logger.info("Tool execution complete: %d tools used", len(tools_used))
        return {"tool_result": tool_result}

    # ── Internal tool dispatch ───────────────────────────────────────────

    async def _invoke_tool(self, name: str, input_data: str) -> Any:
        """Route a tool invocation to the correct handler."""
        if name == "web_search":
            tool: WebSearchTool = self._tools[name]
            results = await tool.search(input_data)
            return tool.format_results(results)

        elif name == "knowledge_retrieval":
            tool_k: KnowledgeTool = self._tools[name]
            results = await tool_k.retrieve(input_data)
            return tool_k.format_results(results)

        elif name == "code_execution":
            tool_c: CodeExecutionTool = self._tools[name]
            result = await tool_c.execute(input_data)
            if result["success"]:
                return f"Output:\n{result['output']}" if result["output"] else "Code executed successfully (no output)."
            else:
                return f"Error:\n{result['error']}"

        raise ValueError(f"No handler for tool: {name}")
