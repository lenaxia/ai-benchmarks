"""Blocking subagent delegation (the ``task`` tool pattern)."""

from __future__ import annotations

import json
from typing import Any

from aibench.agentic.tools.base import Tool


class SubtaskTool(Tool):
    name = "subtask"
    description = (
        "Delegate a self-contained investigation or sub-job to a subagent "
        "(read/glob/grep/bash tools, plus write/edit when agent='general'). "
        "Runs synchronously and returns the subagent's final report. Use for "
        "codebase exploration or independent subproblems; the prompt must be "
        "fully self-contained."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "3-5 word summary"},
            "prompt": {"type": "string", "description": "Complete, self-contained instructions"},
            "agent": {
                "type": "string",
                "enum": ["explore", "general"],
                "description": "explore = read-only research; general = may write",
            },
        },
        "required": ["prompt"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        if not ctx.can_delegate:
            return json.dumps(
                {"error": f"max delegation depth {ctx.max_depth} reached; do it yourself"}
            )
        from aibench.agentic.context import READONLY_SYSTEM
        from aibench.agentic.loop import drive_subagent

        agent = args.get("agent", "explore")
        toolset = "readonly" if agent == "explore" else "plain"
        prompt = args["prompt"]
        child = ctx.child(session_id=f"{ctx.session_id}/subtask", toolset=toolset)
        final_text, _messages = await drive_subagent(
            ctx.provider,
            prompt=prompt,
            system=READONLY_SYSTEM,
            ctx=child,
        )
        report = final_text.strip() or "(subagent returned no text)"
        return f"[subtask:{agent} report]\n{report}"
