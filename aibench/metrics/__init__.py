"""Streaming timing capture: TTFT, inter-token latency, decode throughput."""

from __future__ import annotations

import math
import time

from aibench.types import Timing


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class StreamTracker:
    """Feed chunk timestamps as tokens arrive; snapshot derived timings.

    Feed *content* chunk timestamps (reasoning-only chunks don't count as
    output tokens for TTFT purposes on most serving stacks; we track the
    first visible content/tool-call fragment as first token).
    """

    def __init__(self) -> None:
        self.request_sent = time.perf_counter()
        self.chunk_times: list[float] = []

    def on_token(self) -> None:
        self.chunk_times.append(time.perf_counter())

    def snapshot(self) -> Timing:
        t = Timing(request_sent=self.request_sent, n_chunks=len(self.chunk_times))
        if self.chunk_times:
            t.first_token = self.chunk_times[0]
            t.last_token = self.chunk_times[-1]
            t.ttft_ms = (self.chunk_times[0] - self.request_sent) * 1000.0
        t.total_ms = (time.perf_counter() - self.request_sent) * 1000.0
        if len(self.chunk_times) >= 2:
            itls = [
                (self.chunk_times[i] - self.chunk_times[i - 1]) * 1000.0
                for i in range(1, len(self.chunk_times))
            ]
            t.itl_mean_ms = sum(itls) / len(itls)
            t.itl_p50_ms = percentile(itls, 50)
            t.itl_p95_ms = percentile(itls, 95)
            t.itl_max_ms = max(itls)
        return t


def decode_tps(timing: Timing, output_tokens: int | None) -> float | None:
    """Decode throughput: tokens emitted after the first, per second of decode.

    Uses the inter-token window (last - first) rather than total wall time so
    prefill latency does not pollute decode speed.
    """
    if not output_tokens or output_tokens < 2 or not timing.first_token or not timing.last_token:
        return None
    decode_s = timing.last_token - timing.first_token
    if decode_s <= 0:
        return None
    return (output_tokens - 1) / decode_s


def prefill_tps(timing: Timing, prompt_tokens: int | None, cached_tokens: int | None = None) -> float | None:
    """Approximate prefill throughput: non-cached prompt tokens per second of TTFT.

    Example: 5000 non-cached prompt tokens with a 1000 ms TTFT => 5000 tok/s.
    """
    if not prompt_tokens or timing.ttft_ms is None or timing.ttft_ms <= 0:
        return None
    effective = prompt_tokens - (cached_tokens or 0)
    if effective <= 0:
        return None
    return effective / (timing.ttft_ms / 1000.0)
