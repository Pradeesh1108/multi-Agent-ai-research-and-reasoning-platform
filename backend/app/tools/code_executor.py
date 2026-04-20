"""
Safe Python code execution tool.

Uses AST-based static analysis to block dangerous operations before
running code in a restricted exec() sandbox with captured stdout.
"""

from __future__ import annotations

import ast
import asyncio
import io
import logging
import math
import traceback
from typing import Any

logger = logging.getLogger(__name__)

# ── Deny lists ───────────────────────────────────────────────────────────────

_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "importlib", "socket",
    "requests", "urllib", "http", "ftplib", "smtplib", "ctypes",
    "signal", "multiprocessing", "threading", "pathlib", "glob",
    "tempfile", "pickle", "shelve", "marshal",
})

_BLOCKED_BUILTINS = frozenset({
    "eval", "exec", "__import__", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
})

_BLOCKED_ATTRS = frozenset({
    "__subclasses__", "__bases__", "__class__", "__mro__",
    "__globals__", "__code__", "__closure__",
})

# ── Safe builtins exposed to the sandbox ─────────────────────────────────────

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "hex": hex, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
    "print": print,  # will be redirected to StringIO
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "math": math,
}


class CodeExecutionTool:
    """Executes Python code in a restricted sandbox.

    Parameters:
        timeout: Maximum seconds allowed for code execution.
    """

    def __init__(self, timeout: int = 5) -> None:
        self._timeout = timeout

    # ── Public API ───────────────────────────────────────────────────────

    async def execute(self, code: str) -> dict[str, Any]:
        """Validate and execute a Python code snippet.

        Returns:
            A dict with keys ``success``, ``output``, and ``error``.
        """
        logger.info("Code execution requested (%d chars)", len(code))

        # 1. Static analysis
        safety_issues = self._analyse_safety(code)
        if safety_issues:
            msg = "Code blocked by safety analysis:\n• " + "\n• ".join(safety_issues)
            logger.warning(msg)
            return {"success": False, "output": "", "error": msg}

        # 2. Execute with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_sandboxed, code),
                timeout=self._timeout,
            )
            return result
        except asyncio.TimeoutError:
            msg = f"Code execution timed out after {self._timeout}s"
            logger.warning(msg)
            return {"success": False, "output": "", "error": msg}
        except Exception as exc:
            logger.error("Unexpected execution error: %s", exc, exc_info=True)
            return {"success": False, "output": "", "error": str(exc)}

    # ── Static analysis ──────────────────────────────────────────────────

    def _analyse_safety(self, code: str) -> list[str]:
        """Parse the code's AST and check for blocked constructs."""
        issues: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            issues.append(f"Syntax error: {exc}")
            return issues

        for node in ast.walk(tree):
            # Block dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in _BLOCKED_MODULES:
                        issues.append(f"Blocked import: '{alias.name}'")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module in _BLOCKED_MODULES:
                        issues.append(f"Blocked import from: '{node.module}'")

            # Block dangerous function calls
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
                    issues.append(f"Blocked builtin call: '{func.id}()'")
                elif isinstance(func, ast.Attribute) and func.attr in _BLOCKED_BUILTINS:
                    issues.append(f"Blocked attribute call: '.{func.attr}()'")

            # Block dangerous attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr in _BLOCKED_ATTRS:
                    issues.append(f"Blocked attribute access: '.{node.attr}'")

            # Block open() with write mode
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    # Check for write mode in args
                    if len(node.args) >= 2:
                        mode_arg = node.args[1]
                        if isinstance(mode_arg, ast.Constant) and "w" in str(mode_arg.value):
                            issues.append("Blocked: open() with write mode")

        return issues

    # ── Sandbox execution ────────────────────────────────────────────────

    def _run_sandboxed(self, code: str) -> dict[str, Any]:
        """Execute code in a restricted namespace with captured stdout."""
        stdout_capture = io.StringIO()

        # Build a namespace with safe builtins and redirected print
        sandbox_globals: dict[str, Any] = {
            "__builtins__": {
                **_SAFE_BUILTINS,
                "print": lambda *args, **kwargs: print(
                    *args, **kwargs, file=stdout_capture
                ),
            },
        }
        sandbox_locals: dict[str, Any] = {}

        try:
            exec(code, sandbox_globals, sandbox_locals)  # noqa: S102
            output = stdout_capture.getvalue()
            return {"success": True, "output": output.strip(), "error": None}
        except Exception:
            tb = traceback.format_exc()
            output = stdout_capture.getvalue()
            return {
                "success": False,
                "output": output.strip(),
                "error": tb,
            }
