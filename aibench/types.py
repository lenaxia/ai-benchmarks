"""Core wire types shared across providers, metrics and the agentic loop.

The internal message format follows the OpenAI chat-completions shape
(``role``/``content``/``tool_calls``/``tool_call_id``).  Providers translate
to/from their native wire format; OpenAI-compatible endpoints use it verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]
ToolSchema = dict[str, Any]


@dataclass
class Usage:
    """Token usage reported by the server (when available)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai(cls, usage: dict[str, Any] | None) -> Usage | None:
        if not usage:
            return None
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cached = prompt_details.get("cached_tokens")
        if cached is None:
            # llama.cpp / misc compat spellings
            cached = usage.get("cached_tokens") or usage.get("prompt_cache_hit_tokens")
        reasoning = completion_details.get("reasoning_tokens")
        return cls(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            raw=usage,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class ToolCall:
    """A single function call requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON text, may be malformed; the loop handles errors

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Timing:
    """Wall-clock timing captured from a streaming response."""

    request_sent: float = 0.0
    first_token: float | None = None
    last_token: float | None = None
    n_chunks: int = 0
    # derived
    ttft_ms: float | None = None
    ttft_content_ms: float | None = None
    total_ms: float | None = None
    itl_mean_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_max_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms,
            "ttft_content_ms": self.ttft_content_ms,
            "total_ms": self.total_ms,
            "itl_mean_ms": self.itl_mean_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_p95_ms": self.itl_p95_ms,
            "itl_max_ms": self.itl_max_ms,
            "n_chunks": self.n_chunks,
        }


@dataclass
class ChatResponse:
    """Normalized result of one chat completion (always streamed)."""

    text: str = ""
    reasoning_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage | None = None
    timing: Timing = field(default_factory=Timing)
    error: str | None = None
    retry_wait_s: float = 0.0  # infra backoff time, excluded from scenario budgets
    completion_tokens_observed: int = 0  # fallback count from stream chunks
    prompt_tokens_estimated: int | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def output_tokens(self) -> int | None:
        if self.usage and self.usage.completion_tokens is not None:
            return self.usage.completion_tokens
        return self.completion_tokens_observed or None

    def prompt_tokens(self) -> int | None:
        if self.usage and self.usage.prompt_tokens is not None:
            return self.usage.prompt_tokens
        return self.prompt_tokens_estimated
