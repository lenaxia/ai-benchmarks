"""Execute one scenario: build the world, run the agent loop, grade checks,
capture per-turn metrics (TTFT, cached tokens across the loop, tool usage).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from aibench.agentic.board import BoardStore
from aibench.agentic.context import (
    DEFAULT_SYSTEM_PLAIN,
    DEFAULT_SYSTEM_SAFESPACE,
    RunContext,
)
from aibench.agentic.env import Env
from aibench.agentic.loop import run_agent_loop
from aibench.agentic.registry import build_registry
from aibench.agentic.scenarios.base import CheckContext, CheckResult, Scenario, SetupBundle
from aibench.agentic.sessions import SessionPlatform
from aibench.providers.base import Provider


@dataclass
class ScenarioResult:
    scenario: str
    passed: bool = False
    score: float = 0.0
    checks: list[CheckResult] = field(default_factory=list)
    stop_reason: str = ""
    final_text: str = ""
    elapsed_s: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    board_state: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "passed": self.passed,
            "score": self.score,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
            "stop_reason": self.stop_reason,
            "final_text": self.final_text[:2000],
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "metrics": self.metrics,
            "board_state": self.board_state,
        }


async def run_scenario(
    scenario: Scenario,
    provider: Provider,
    *,
    subagent_max_turns: int = 12,
    max_depth: int = 1,
    keep_env: bool = False,
    deterministic_ids: bool = False,
) -> ScenarioResult:
    env = Env()
    board = BoardStore(deterministic_ids=deterministic_ids)
    sessions = SessionPlatform()
    ctx = RunContext(
        env=env,
        board=board,
        provider=provider,
        sessions=sessions,
        max_tool_calls=scenario.max_tool_calls,
        subagent_max_turns=subagent_max_turns,
        max_depth=max_depth,
        toolset=scenario.toolset,
    )
    result = ScenarioResult(scenario=scenario.name)
    try:
        data: dict[str, Any] = {"session_id": ctx.session_id}
        if scenario.setup:
            bundle = SetupBundle(env, board, sessions, data, session_id=ctx.session_id)
            scenario.setup(bundle)
        task_text = scenario.render_task(data)

        ctx.deadline = time.time() + scenario.max_seconds
        registry = build_registry(scenario.toolset, ctx)
        system = scenario.system or (
            DEFAULT_SYSTEM_PLAIN
            if scenario.toolset == "plain"
            else DEFAULT_SYSTEM_SAFESPACE
        )

        t0 = time.time()
        loop = await run_agent_loop(
            provider,
            system=system,
            prompt=task_text,
            registry=registry,
            ctx=ctx,
            max_turns=scenario.max_turns,
        )
        # give background sessions a bounded chance to finish
        await sessions.wait_all(timeout_s=max(30.0, scenario.max_seconds / 4))
        result.elapsed_s = time.time() - t0

        check_ctx = CheckContext(
            env=env,
            board=board,
            sessions=sessions,
            run_ctx=ctx,
            loop_result=loop,
            data=data,
            elapsed_s=result.elapsed_s,
        )
        checks = list(scenario.checks)
        checks += [
            ("budget_tool_calls", _check_budget(scenario)),
            ("wall_time", _check_wall(scenario)),
        ]
        for name, fn in checks:
            try:
                ok, detail = await fn(check_ctx) if asyncio.iscoroutinefunction(fn) else fn(check_ctx)
            except Exception as e:  # noqa: BLE001 — a broken check must not sink the run
                ok, detail = False, f"check crashed: {type(e).__name__}: {e}"
            result.checks.append(CheckResult(name=name, passed=ok, detail=str(detail)[:500]))

        result.score = (
            sum(1 for c in result.checks if c.passed) / len(result.checks)
            if result.checks
            else 0.0
        )
        result.passed = all(c.passed for c in result.checks)
        result.stop_reason = loop.stop_reason
        result.final_text = loop.final_text
        result.metrics = ctx.metrics.to_dict()
        result.board_state = board.state()
        result.transcript = loop.messages
    except Exception as e:  # noqa: BLE001
        result.error = f"{type(e).__name__}: {e}"
    finally:
        if not keep_env:
            env.cleanup()
    return result


def _check_budget(scenario: Scenario):
    def _fn(ctx: CheckContext) -> tuple[bool, str]:
        n = ctx.run_ctx.metrics.tool_calls
        return n <= scenario.max_tool_calls, f"{n}/{scenario.max_tool_calls} tool calls"

    return _fn


def _check_wall(scenario: Scenario):
    def _fn(ctx: CheckContext) -> tuple[bool, str]:
        ok = ctx.elapsed_s <= scenario.max_seconds
        return ok, f"{ctx.elapsed_s:.0f}s/{scenario.max_seconds:.0f}s"

    return _fn
