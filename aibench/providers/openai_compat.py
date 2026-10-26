"""OpenAI-compatible chat-completions provider (vLLM, llama.cpp server,
LM Studio, TGI, SGLang, OpenRouter, OpenAI itself, ...).

Speaks raw HTTP/SSE via httpx rather than the openai SDK so we control
timing capture exactly and stay compatible with the widest set of servers.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import httpx

from aibench.metrics import StreamTracker
from aibench.providers.base import Provider, register_provider
from aibench.types import ChatResponse, Message, Timing, ToolCall, ToolSchema, Usage


class _Trackers:
    """Three views of the same stream.

    ``any``      — every output delta (reasoning + content + tool fragments);
                   the standard TTFT timeline and decode-span source, correct
                   for reasoning models where visible content arrives late.
    ``content``  — visible content only (user-latency view).
    ``reasoning``— reasoning deltas only.
    """

    def __init__(self) -> None:
        self.any = StreamTracker()
        self.content = StreamTracker()
        self.reasoning = StreamTracker()


    def snapshot(self) -> Timing:
        t = self.any.snapshot()
        content_snap = self.content.snapshot()
        t.ttft_content_ms = content_snap.ttft_ms
        return t


class _Retryable(Exception):
    pass


def _cooldown_hint(e: Exception) -> float | None:
    import re

    text = str(e)
    m = re.search(r"[Tt]ry again in (\d+) seconds", text)
    if m:
        return float(m.group(1))
    m = re.search(r"Retry-After:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


class _Fatal(Exception):
    pass


class _BadRequest(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@register_provider("openai")
class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 300.0,
        max_retries: int = 3,
        extra_headers: dict[str, str] | None = None,
        stream_options: bool = True,
        request_delay_s: float = 0.0,
        extra_body: dict[str, Any] | None = None,
        max_cooldown_wait_s: float = 320.0,
    ) -> None:
        self.extra_body = extra_body or {}
        self.max_cooldown_wait_s = max_cooldown_wait_s
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.stream_options = stream_options
        self.request_delay_s = request_delay_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
        )

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if self.extra_body:
            payload.update(self.extra_body)
        if extra_body:
            payload.update(extra_body)

        attempt = 0
        retry_wait = 0.0
        use_stream_options = self.stream_options
        last_err: Exception | None = None
        while attempt <= self.max_retries:
            if self.request_delay_s:
                await asyncio.sleep(self.request_delay_s)
            trackers = _Trackers()
            try:
                return await self._stream_once(payload, trackers, use_stream_options)
            except _BadRequest as e:
                # Some servers reject stream_options; retry once without it.
                if use_stream_options and "stream_options" in str(e.detail):
                    use_stream_options = False
                    continue
                return ChatResponse(error=f"HTTP 400: {e.detail}")
            except _Fatal as e:
                return ChatResponse(error=f"request failed: {e}")
            except (TimeoutError, _Retryable, httpx.HTTPError) as e:
                last_err = e
                attempt += 1
                if attempt > self.max_retries:
                    break
                backoff = min(2.0**attempt, 30.0) + random.uniform(0, 0.5)
                # honor server cooldown hints ("Try again in 300 seconds",
                # Retry-After) so pool-wide 429s don't burn the retry budget
                # against a wall
                hint = _cooldown_hint(e)
                if hint is not None:
                    backoff = max(backoff, min(hint, self.max_cooldown_wait_s))
                retry_wait += backoff
                await asyncio.sleep(backoff)
        return ChatResponse(
            error=f"request failed after {attempt + 1} attempts: {last_err!r}",
            retry_wait_s=retry_wait,
        )

    async def _stream_once(
        self,
        payload: dict[str, Any],
        trackers: _Trackers,
        use_stream_options: bool,
    ) -> ChatResponse:
        body = dict(payload)
        if use_stream_options:
            body["stream_options"] = {"include_usage": True}
        resp = await self._client.post("/chat/completions", json=body)
        if resp.status_code == 400:
            raise _BadRequest(resp.text[:2000])
        if resp.status_code == 429 or resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:500]}")
        if resp.status_code != 200:
            raise _Fatal(f"HTTP {resp.status_code}: {resp.text[:500]}")

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage: Usage | None = None

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue  # keepalive / comment
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise _Fatal(str(chunk["error"])[:500])
            if chunk.get("usage"):
                usage = Usage.from_openai(chunk["usage"])
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                text_parts.append(content)
                trackers.content.on_token()
                trackers.any.on_token()
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                reasoning_parts.append(reasoning)
                trackers.reasoning.on_token()
                trackers.any.on_token()
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                acc = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["arguments"] += fn["arguments"]
                trackers.any.on_token()

        timing = trackers.snapshot()
        calls = [
            ToolCall(
                id=acc.get("id") or f"call_{i}",
                name=acc.get("name", ""),
                arguments=acc.get("arguments", ""),
            )
            for i, acc in sorted(tool_calls.items())
        ]
        return ChatResponse(
            text="".join(text_parts),
            reasoning_text="".join(reasoning_parts),
            tool_calls=calls,
            finish_reason=finish_reason,
            usage=usage,
            timing=timing,
            completion_tokens_observed=timing.n_chunks,
        )

    async def close(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAICompatProvider"]
