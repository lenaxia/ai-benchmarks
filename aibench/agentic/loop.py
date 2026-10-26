"""The agentic tool loop: model turns, tool execution, budgets, per-turn
metrics (including cached-token accounting — agentic loops re-send the whole
conversation, so prefix-cache behaviour is a first-class measurement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aibench.agentic.tools.base import ToolRegistry
from aibench.types import ChatResponse, Message

if TYPE_CHECKING:
    from aibench.agentic.context import RunContext
    from aibench.providers.base import Provider


@dataclass
class LoopResult:
    final_text: str = ""
    messages: list[Message] = field(default_factory=list)
    stop_reason: str = "completed"  # completed|max_turns|budget|deadline|error
    turns: int = 0


def _assistant_message(resp: ChatResponse) -> Message:
    msg: Message = {"role": "assistant", "content": resp.text if resp.text else None}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]
    return msg


def _record_turn(ctx: RunContext, resp: ChatResponse, tag: str, turn: int) -> None:
    usage = resp.usage
    row = {
        "depth": ctx.depth,
        "tag": tag,
        "turn": turn,
        "ttft_ms": resp.timing.ttft_ms,
        "total_ms": resp.timing.total_ms,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "cached_tokens": usage.cached_tokens if usage else None,
        "reasoning_tokens": usage.reasoning_tokens if usage else None,
        "tool_calls": len(resp.tool_calls),
    }
    ctx.metrics.per_turn.append(row)
    if usage:
        ctx.metrics.prompt_tokens += usage.prompt_tokens or 0
        ctx.metrics.completion_tokens += usage.completion_tokens or 0
        ctx.metrics.cached_tokens += usage.cached_tokens or 0
        ctx.metrics.reasoning_tokens += usage.reasoning_tokens or 0


async def run_agent_loop(
    provider: Provider,
    *,
    system: str,
    prompt: str | None = None,
    history: list[Message] | None = None,
    registry: ToolRegistry,
    ctx: RunContext,
    max_turns: int = 40,
    tag: str = "main",
) -> LoopResult:
    """Run one agent conversation until it stops calling tools or hits a limit.

    Either ``prompt`` (fresh conversation) or ``history`` (continuation, in
    which case ``prompt`` is appended as the next user message) must be given.
    """
    if history:
        messages = list(history)
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt or ""},
        ]

    result = LoopResult(messages=messages)
    for turn in range(1, max_turns + 1):
        if not ctx.budget_left():
            result.stop_reason = "budget"
            break
        result.turns = turn
        ctx.metrics.turns += 1
        resp = await provider.chat(messages=messages, tools=registry.schemas())
        _record_turn(ctx, resp, tag, turn)

        if resp.retry_wait_s:
            # infra waits must not consume the scenario's wall budget
            ctx.deadline = (ctx.deadline or 0) + resp.retry_wait_s
        if not resp.ok:
            result.stop_reason = "error"
            messages.append(
                {"role": "assistant", "content": f"[provider error: {resp.error}]"}
            )
            break

        messages.append(_assistant_message(resp))
        for tc in resp.tool_calls:
            ctx.metrics.by_tool[tc.name] = ctx.metrics.by_tool.get(tc.name, 0) + 1
            outcome = await registry.execute(tc.name, tc.arguments, ctx)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": outcome.output,
                }
            )
            if not ctx.budget_left():
                result.stop_reason = "budget"
                return _finish(result, messages, ctx)

        if not resp.tool_calls:
            result.final_text = resp.text
            result.stop_reason = "completed"
            return _finish(result, messages, ctx)

    else:
        result.stop_reason = "max_turns"
    return _finish(result, messages, ctx)


def _finish(result: LoopResult, messages: list[Message], ctx: RunContext) -> LoopResult:
    result.messages = messages
    return result


async def drive_subagent(
    provider: Provider | None = None,
    *,
    prompt: str,
    system: str | None,
    ctx: RunContext,
    history: list[Message] | None = None,
    registry: ToolRegistry | None = None,
) -> tuple[str, list[Message]]:
    """Loop driver used by session_create / send_message / subtask.

    Builds its own registry for the child toolset and runs a bounded loop.
    ``provider`` defaults to the one on ``ctx`` (shared benchmark model).
    """
    from aibench.agentic.registry import build_registry

    reg = registry or build_registry(ctx.toolset, ctx)
    res = await run_agent_loop(
        provider or ctx.provider,
        system=system or "",
        prompt=prompt,
        history=history,
        registry=reg,
        ctx=ctx,
        max_turns=ctx.subagent_max_turns,
        tag=f"session:{ctx.session_id}",
    )
    return res.final_text, res.messages
