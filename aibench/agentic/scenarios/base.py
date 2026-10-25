"""Scenario model: setup, task, deterministic checks, budgets.

Scenarios are the benchmark unit for agentic loops. ``setup`` seeds the
sandbox/board/sessions and fills a ``data`` dict; ``task`` is rendered with
``{placeholders}`` from that dict (charter envelope style); ``checks`` are
pure functions over the final world state.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aibench.agentic.board import BoardStore
from aibench.agentic.env import Env
from aibench.agentic.sessions import SessionPlatform


class SetupBundle:
    def __init__(
        self,
        env: Env,
        board: BoardStore,
        sessions: SessionPlatform,
        data: dict[str, Any],
        session_id: str = "ses-main",
    ):
        self.env = env
        self.board = board
        self.sessions = sessions
        self.data = data
        self.session_id = session_id


@dataclass
class CheckContext:
    env: Env
    board: BoardStore
    sessions: SessionPlatform
    run_ctx: Any  # RunContext (untyped to avoid import cycle)
    loop_result: Any  # LoopResult
    data: dict[str, Any]
    elapsed_s: float


CheckFn = Callable[[CheckContext], tuple[bool, str]]


@dataclass
class Scenario:
    name: str
    title: str
    toolset: str  # plain | safespace
    task: str
    system: str | None = None
    setup: Callable[[SetupBundle], None] | None = None
    checks: list[tuple[str, CheckFn]] = field(default_factory=list)
    max_turns: int = 40
    max_tool_calls: int = 120
    max_seconds: float = 600.0

    def render_task(self, data: dict[str, Any]) -> str:
        """Replace {key} placeholders for known keys only (safe against braces)."""

        def sub(m: re.Match) -> str:
            key = m.group(1)
            return str(data[key]) if key in data else m.group(0)

        return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", sub, self.task)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def audit_events(task_row: Any, event_prefix: str) -> list[dict]:
    return [a for a in task_row.audit if a["event"].startswith(event_prefix)]


def tool_calls_named(run_ctx: Any, name: str) -> list[Any]:
    return [c for c in run_ctx.env.call_log if c.name == name and c.depth == 0]
