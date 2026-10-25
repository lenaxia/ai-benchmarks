"""llmsafespaces-flavoured toolset, mocked in-process.

Board tools wrap :class:`aibench.agentic.board.BoardStore`; session tools wrap
:class:`aibench.agentic.sessions.SessionPlatform` and drive *real* nested
agent loops with the benchmarked model. BoardError payloads surface as JSON
with status/code/message exactly like the platform API the agents we studied
learned to handle (409 claim_token, 400 future_stamp, 409 criteria_outstanding...).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from aibench.agentic.board import BoardError
from aibench.agentic.tools.base import Tool


def _dump(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=None)


def _make_board_tool(name: str, description: str, parameters: dict, handler) -> type[Tool]:  # type: ignore[no-untyped-def]
    async def _run(self, args: dict[str, Any], ctx) -> str:  # type: ignore[no-untyped-def]
        try:
            return _dump(handler(ctx.board, args, ctx))
        except BoardError as e:
            ctx.metrics.board_rejections.append(e.code)
            return _dump(e.payload())

    return type(
        f"BoardTool_{name}",
        (Tool,),
        {
            "name": name,
            "description": description,
            "parameters": parameters,
            "run": _run,
        },
    )


BoardCreateProject = _make_board_tool(
    "board_create_project",
    "Create a Task Board project (swimlane for structured multi-agent work).",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "worker_slots": {"type": "integer"},
        },
        "required": ["name"],
    },
    lambda board, a, ctx: board.create_project(a["name"], int(a.get("worker_slots", 2))),
)

BoardCreateEpic = _make_board_tool(
    "board_create_epic",
    "Create an epic in a project.",
    {
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "title": {"type": "string"},
            "outcome": {"type": "string"},
        },
        "required": ["project_id", "title"],
    },
    lambda board, a, ctx: board.create_epic(a["project_id"], a["title"], a.get("outcome", "")),
)

BoardCreateTask = _make_board_tool(
    "board_create_task",
    "Create a backlog task in an epic with acceptance criteria and optional dependencies.",
    {
        "type": "object",
        "properties": {
            "epic_id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Checkable criteria that define Done",
            },
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["epic_id", "title"],
    },
    lambda board, a, ctx: board.create_task(
        a["epic_id"],
        a["title"],
        a.get("description", ""),
        a.get("criteria"),
        a.get("depends_on"),
    ),
)

BoardClaim = _make_board_tool(
    "board_claim",
    "Claim a backlog task for this session. Returns the charter envelope with "
    "claim_token and lease_expires_at. Your session is bound to the claim.",
    {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    },
    lambda board, a, ctx: board.claim(a["task_id"], ctx.session_id),
)

BoardHeartbeat = _make_board_tool(
    "board_heartbeat",
    "Renew your claimed task's lease. observed_progress_at (unix seconds) must "
    "be a real current timestamp strictly newer than your last one; call "
    "get_datetime first. Future stamps are rejected.",
    {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "claim_token": {"type": "string"},
            "observed_progress_at": {"type": "number"},
        },
        "required": ["task_id", "claim_token", "observed_progress_at"],
    },
    lambda board, a, ctx: board.heartbeat(
        a["task_id"], a["claim_token"], float(a["observed_progress_at"])
    ),
)

BoardMoveStage = _make_board_tool(
    "board_move_stage",
    "Move your task's stage. in_progress->in_review needs your claim_token and "
    "an evidenceUrl. in_review->done is refused until all criteria are met.",
    {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "to_stage": {"type": "string", "enum": ["in_progress", "in_review", "done", "pending_input"]},
            "claim_token": {"type": "string"},
            "evidence_url": {"type": "string"},
        },
        "required": ["task_id", "to_stage"],
    },
    lambda board, a, ctx: board.move_stage(
        a["task_id"],
        a["to_stage"],
        a.get("claim_token"),
        a.get("evidence_url"),
    ),
)

BoardUpdateTask = _make_board_tool(
    "board_update_task",
    "Patch a task: set description, or mark a criterion met by its 0-based "
    "index (set_criteria_met). Only mark criteria met when evidence is verified.",
    {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "description": {"type": "string"},
            "title": {"type": "string"},
            "set_criteria_met": {"type": "integer"},
        },
        "required": ["task_id"],
    },
    lambda board, a, ctx: board.update_task(
        a["task_id"],
        description=a.get("description"),
        set_criteria_met=a.get("set_criteria_met"),
    ),
)

BoardRead = _make_board_tool(
    "board_read",
    "Read a task's full state: stage, criteria, blocked-by dependencies, "
    "claim/lease, audit trail.",
    {"type": "object", "properties": {"task_id": {"type": "string"}},
     "required": ["task_id"]},
    lambda board, a, ctx: board.read(a["task_id"]),
)

BoardState = _make_board_tool(
    "board_list",
    "List the whole board: projects, epics, tasks with stages (manager view).",
    {"type": "object", "properties": {}},
    lambda board, a, ctx: board.state(),
)


class GetDatetimeTool(Tool):
    name = "get_datetime"
    description = (
        "Current date/time: ISO-8601 UTC and unix epoch seconds. Call this "
        "before board_heartbeat to get a valid observed_progress_at."
    )
    parameters = {"type": "object", "properties": {}}

    async def run(self, args: dict[str, Any], ctx) -> str:
        now = datetime.now(UTC)
        return _dump(
            {
                "iso": now.isoformat(),
                "epoch": now.timestamp(),
                "board_epoch": ctx.board.now(),
                "note": "use epoch for board_heartbeat observed_progress_at",
            }
        )


class _SessionTool(Tool):
    def __init__(self, platform_holder) -> None:  # type: ignore[no-untyped-def]
        self._platform = platform_holder


class SessionCreateTool(_SessionTool):
    name = "session_create"
    description = (
        "Spawn a NEW agent session (fire-and-forget) with a self-contained "
        "first prompt. It runs in the background sharing this workspace/board. "
        "Returns immediately with the session id."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "prompt": {"type": "string"},
            "toolset": {
                "type": "string",
                "enum": ["plain", "safespace"],
                "description": "plain = coding tools; safespace = board+session tools",
            },
        },
        "required": ["prompt"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        from aibench.agentic.context import DEFAULT_SYSTEM_PLAIN, DEFAULT_SYSTEM_SAFESPACE
        from aibench.agentic.loop import drive_subagent

        toolset = args.get("toolset", "plain")
        system = (
            DEFAULT_SYSTEM_SAFESPACE if toolset == "safespace" else DEFAULT_SYSTEM_PLAIN
        )
        info = await self._platform().create_session(
            args.get("title", "worker"),
            args["prompt"],
            toolset=toolset,
            driver=drive_subagent,
            system=system,
            ctx=ctx,
        )
        return _dump(info)


class SessionListTool(_SessionTool):
    name = "session_list"
    description = "List all sessions in this workspace (id, title, status)."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, platform_holder=lambda: None) -> None:
        super().__init__(platform_holder)

    async def run(self, args: dict[str, Any], ctx) -> str:
        return _dump(self._platform().list_sessions())


class SessionReadTool(_SessionTool):
    name = "session_read"
    description = "Read a session's recent messages (its transcript tail)."
    parameters = {
        "type": "object",
        "properties": {"session_id": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["session_id"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        try:
            data = self._platform().read(args["session_id"], int(args.get("limit", 20)))
        except KeyError as e:
            return _dump({"error": str(e)})
        return _dump(data)


class SessionMetadataTool(_SessionTool):
    name = "session_metadata"
    description = "Session vitals: status, busy flag, message count, inbox."
    parameters = {
        "type": "object",
        "properties": {"session_id": {"type": "string"}},
        "required": ["session_id"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        try:
            return _dump(self._platform().metadata(args["session_id"]))
        except KeyError as e:
            return _dump({"error": str(e)})


class SendMessageTool(_SessionTool):
    name = "send_message"
    description = (
        "Send a text message to another session (fire-and-forget). If the "
        "target is busy it is queued and delivered when its current turn ends."
    )
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["session_id", "message"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        from aibench.agentic.context import DEFAULT_SYSTEM_PLAIN
        from aibench.agentic.loop import drive_subagent

        try:
            result = await self._platform().send_message(
                args["session_id"],
                args["message"],
                from_id=ctx.session_id,
                driver=drive_subagent,
                system=DEFAULT_SYSTEM_PLAIN,
                ctx=ctx,
            )
        except KeyError as e:
            return _dump({"error": str(e)})
        ctx.metrics.sent_messages.append(
            {"to": args["session_id"], "text": args["message"][:500]}
        )
        return _dump(result)


def safespace_tools(platform_holder) -> list[Tool]:  # type: ignore[no-untyped-def]
    return [
        BoardCreateProject(),
        BoardCreateEpic(),
        BoardCreateTask(),
        BoardClaim(),
        BoardHeartbeat(),
        BoardMoveStage(),
        BoardUpdateTask(),
        BoardRead(),
        BoardState(),
        GetDatetimeTool(),
        SessionCreateTool(platform_holder),
        SessionListTool(),
        SessionReadTool(platform_holder),
        SessionMetadataTool(platform_holder),
        SendMessageTool(platform_holder),
    ]
