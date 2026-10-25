"""Tool abstraction + registry with malformed-argument handling.

A tool receives parsed JSON args plus the run context and returns a string
(the ``role: tool`` message content). Errors are returned as structured
JSON strings so the model can see and recover from them — recovery is one
of the things we benchmark.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aibench.agentic.context import RunContext


class Tool(ABC):
    name: str = "tool"
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: RunContext) -> str: ...

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallOutcome:
    output: str
    ok: bool


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def names(self) -> list[str]:
        return sorted(self.tools)

    async def execute(self, name: str, raw_arguments: str, ctx: RunContext) -> ToolCallOutcome:
        import time

        from aibench.agentic.env import CallRecord

        start = time.time()
        output: str
        ok = True
        try:
            if name not in self.tools:
                output = _err(
                    f"unknown tool {name!r}; available: {', '.join(self.names())}"
                )
                ok = False
            else:
                try:
                    args = json.loads(raw_arguments) if raw_arguments.strip() else {}
                except json.JSONDecodeError as e:
                    output = _err(f"malformed JSON arguments for {name}: {e}")
                    ok = False
                    ctx.metrics.arg_parse_failures += 1
                else:
                    if not isinstance(args, dict):
                        output = _err(f"arguments for {name} must be a JSON object")
                        ok = False
                    else:
                        output = await self.tools[name].run(args, ctx)
        except Exception as e:  # tool bugs must not kill the loop
            output = _err(f"tool {name} crashed: {type(e).__name__}: {e}")
            ok = False
        ctx.env.call_log.append(
            CallRecord(ctx.depth, name, _safe_args(raw_arguments), ok, output, time.time() - start)
        )
        ctx.metrics.tool_calls += 1
        return ToolCallOutcome(output=output, ok=ok)


def _err(message: str) -> str:
    return json.dumps({"error": message})


def _safe_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw) if raw.strip() else {}
        if isinstance(parsed, dict):
            # keep logs bounded
            return {
                k: (v[:400] + "..." if isinstance(v, str) and len(v) > 400 else v)
                for k, v in parsed.items()
            }
    except json.JSONDecodeError:
        pass
    return {"_raw": raw[:400]}
