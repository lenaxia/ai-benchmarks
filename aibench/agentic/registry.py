"""Toolset assembly: plain / safespace / readonly, plus subtask delegation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aibench.agentic.tools.base import Tool, ToolRegistry
from aibench.agentic.tools.plain import plain_tools
from aibench.agentic.tools.subtask import SubtaskTool

if TYPE_CHECKING:
    from aibench.agentic.context import RunContext


def build_registry(toolset: str, ctx: RunContext) -> ToolRegistry:
    reg = ToolRegistry()
    tools: list[Tool] = []

    if toolset == "plain":
        tools = plain_tools()
    elif toolset == "readonly":
        from aibench.agentic.tools.plain import BashTool, GlobTool, GrepTool, ReadTool

        tools = [ReadTool(), GlobTool(), GrepTool(), BashTool()]
    elif toolset in ("safespace", "both"):
        from aibench.agentic.tools.safespace import safespace_tools

        # the real workspace exposes file tools AND board/session tools together
        tools = plain_tools() + safespace_tools(lambda: ctx.sessions)
    else:
        raise ValueError(f"unknown toolset {toolset!r}")

    for t in tools:
        reg.add(t)
    if ctx.can_delegate and toolset in ("plain", "safespace", "both"):
        reg.add(SubtaskTool())
    return reg
