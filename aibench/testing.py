"""Offline testing helpers.

:class:`ScriptedProvider` replays a pre-written sequence of assistant turns
so the whole loop / tool / board machinery can be exercised deterministically
without a server — useful for CI and for validating custom scenarios.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from aibench.metrics import StreamTracker
from aibench.providers.base import Provider
from aibench.types import ChatResponse, Message, ToolCall, ToolSchema, Usage


@dataclass
class Step:
    text: str = ""
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    # {"name": ..., "arguments": <json-str-or-dict>}
    usage: dict[str, Any] | None = None
    error: str | None = None
    finish_reason: str | None = None


class ScriptedProvider(Provider):
    name = "scripted"

    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []
        self._n = 0

    async def chat(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.calls.append({"messages": messages, "tools": tools, "n": self._n})
        idx = self._n
        self._n += 1
        if idx >= len(self.steps):
            step = Step(text="(scripted end)")
        else:
            step = self.steps[idx]

        tracker = StreamTracker()
        time.sleep(0.001)
        if step.text:
            tracker.on_token()
        usage = Usage.from_openai(step.usage)
        calls = []
        for i, tc in enumerate(step.tool_calls):
            args = tc.get("arguments", "{}")
            if not isinstance(args, str):
                args = json.dumps(args)
            calls.append(ToolCall(id=tc.get("id", f"call_{idx}_{i}"), name=tc["name"], arguments=args))
            tracker.on_token()
        return ChatResponse(
            text=step.text,
            tool_calls=calls,
            finish_reason=step.finish_reason or ("tool_calls" if calls else "stop"),
            usage=usage,
            timing=tracker.snapshot(),
            error=step.error,
            completion_tokens_observed=max(1, len(step.text.split())),
            prompt_tokens_estimated=sum(len(str(m.get("content") or "").split()) for m in messages),
        )

    async def close(self) -> None:  # noqa: D102
        pass
