"""Shared run context threaded through every tool call and nested loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aibench.agentic.board import BoardStore
from aibench.agentic.env import Env
from aibench.agentic.sessions import SessionPlatform
from aibench.providers.base import Provider

DEFAULT_SYSTEM_PLAIN = """\
You are an autonomous coding agent working in a sandboxed workspace.

Tools: read, write, edit, bash, glob, grep (and subtask for delegation).
Work loop: investigate first (read/glob/grep/bash), make the smallest \
change that satisfies the task, then VERIFY by running the code or tests \
with bash before finishing.

Rules:
- Keep total tool calls within budget; avoid redundant repeats.
- If a tool returns an error, read it and adapt — do not repeat the same \
failing call more than twice.
- When the task is done and verified, stop and state the result concisely \
with the evidence (commands run and their outcomes).
"""

DEFAULT_SYSTEM_SAFESPACE = """\
You are an autonomous agent operating inside a workspace with a task board \
(llmsafespaces-style), a multi-session environment, and coding tools \
(read/write/edit/bash/glob/grep).

Board protocol:
- Claim a backlog task with board_claim to get a claim_token and a lease.
- Your claim token is required for in-flight board writes. A 409 error with \
code "claim_token" means your lease lapsed or was taken: STOP writing, \
board_read the task, and either cleanly stand down or re-claim.
- Renew leases with board_heartbeat; observed_progress_at must be a real \
current timestamp (use get_datetime) strictly newer than your last one — \
future stamps are rejected.
- Move work to in_review with board_move_stage plus an evidenceUrl \
(e.g. file://relative/path or a PR URL).
- done is refused until every acceptance criterion is marked met via \
board_update_task(set_criteria_met=...). Only mark criteria met when you \
have verified the evidence yourself.
- session_create spawns a background worker session; steer or reply with \
send_message; inspect with session_list / session_read / session_metadata.
- subtask runs a read-only delegation synchronously and returns its report.

Finish by reporting status concisely with evidence.
"""

READONLY_SYSTEM = """\
You are a read-only research subagent. Investigate using read, glob, grep \
and non-mutating bash only. Do not create, modify or delete anything. \
End with a structured report answering every part of the request, citing \
file paths and quoting the exact lines that matter.
"""


@dataclass
class RunMetrics:
    turns: int = 0
    tool_calls: int = 0
    by_tool: dict[str, int] = field(default_factory=dict)
    arg_parse_failures: int = 0
    board_rejections: list[str] = field(default_factory=list)  # error codes
    sent_messages: list[dict[str, Any]] = field(default_factory=list)
    per_turn: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "by_tool": dict(self.by_tool),
            "arg_parse_failures": self.arg_parse_failures,
            "board_rejections": list(self.board_rejections),
            "sent_messages": self.sent_messages,
            "per_turn": self.per_turn,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


def _next_session_id() -> str:

    _next_session_id.counter = getattr(_next_session_id, "counter", 0) + 1  # type: ignore[attr-defined]
    return f"ses-main-{_next_session_id.counter}"  # type: ignore[attr-defined]


@dataclass
class RunContext:
    env: Env
    board: BoardStore
    provider: Provider
    sessions: SessionPlatform = field(default_factory=SessionPlatform)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    session_id: str = field(default_factory=_next_session_id)
    depth: int = 0
    max_depth: int = 1
    max_tool_calls: int = 250
    subagent_max_turns: int = 12
    deadline: float | None = None
    toolset: str = "plain"
    verbose: bool = False

    def child(self, session_id: str, toolset: str) -> RunContext:
        c = RunContext(
            env=self.env,
            board=self.board,
            provider=self.provider,
            sessions=self.sessions,
            metrics=self.metrics,
            session_id=session_id,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            max_tool_calls=self.max_tool_calls,
            subagent_max_turns=self.subagent_max_turns,
            deadline=self.deadline,
            toolset=toolset,
        )
        return c

    @property
    def can_delegate(self) -> bool:
        return self.depth < self.max_depth

    def budget_left(self) -> bool:
        if self.metrics.tool_calls >= self.max_tool_calls:
            return False
        return self.deadline is None or time.time() <= self.deadline
