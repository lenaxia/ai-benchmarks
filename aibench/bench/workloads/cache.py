"""Prefix-cache behaviour: cold (miss) vs warm (hit) with matched prompts.

Miss phase: every request carries a UUID-salted unique prefix — a serving
stack with prefix caching must show ~0 cached tokens and cold TTFT.

Hit phase: one identical prefix is sent repeatedly with only a trailing
question varying; after the first (warm-up) request, a prefix-caching
server reports cached_tokens ≈ prefix size and TTFT drops sharply.

Speedup = median(cold TTFT) / median(warm TTFT). Both timing-based and
usage-based (cached_tokens) evidence is recorded because not every server
reports the field.
"""

from __future__ import annotations

import statistics

from aibench.bench.corpus import CACHE_PROBE_INSTRUCTION, TokenEstimator
from aibench.bench.workloads.base import RequestRecord, Workload
from aibench.providers.base import Provider


class CacheWorkload(Workload):
    name = "cache"

    def __init__(
        self,
        prefix_tokens: int = 4096,
        estimator: TokenEstimator | None = None,
        temperature: float | None = None,
    ):
        self.prefix_tokens = prefix_tokens
        self.estimator = estimator or TokenEstimator()
        self.temperature = temperature

    def _prefix_body(self, unique: bool, salt: int = 0) -> str:
        # Fixed seed => stable shared prefix for the hit phase.
        return self.estimator.filler(self.prefix_tokens, seed=7000 + salt, unique=unique)

    async def _send(self, provider: Provider, body: str, question: str) -> RequestRecord:
        prompt = (
            "Background material follows; it is reference-only context.\n\n"
            f"{body}\n\n"
            f"{question}"
        )
        resp = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=48,
            temperature=self.temperature,
        )
        return RequestRecord(
            workload=self.name,
            label="",
            target_prompt_tokens=self.prefix_tokens,
            target_output_tokens=48,
            response=resp,
        )

    async def run(self, provider: Provider, repeats: int, warmup: int) -> list[RequestRecord]:
        records: list[RequestRecord] = []

        # ---- miss phase: unique prefix every request ----
        for rep in range(warmup + repeats):
            body = self._prefix_body(unique=True, salt=rep)
            rec = await self._send(provider, body, CACHE_PROBE_INSTRUCTION)
            rec.label = "miss"
            rec.derived["phase"] = "miss"
            records.append(rec)

        # ---- hit phase: shared prefix, tiny unique suffix ----
        shared = self._prefix_body(unique=False)
        first = await self._send(provider, shared, f"{CACHE_PROBE_INSTRUCTION} (reference tag: seed-0)")
        first.label = "hit/warmup"
        first.derived["phase"] = "hit_warmup"
        records.append(first)
        for rep in range(repeats):
            rec = await self._send(
                provider, shared, f"{CACHE_PROBE_INSTRUCTION} (reference tag: seed-{rep + 1})"
            )
            rec.label = "hit"
            rec.derived["phase"] = "hit"
            records.append(rec)

        ttfts = lambda rows: [  # noqa: E731
            r.response.timing.ttft_ms for r in rows if r.response.timing.ttft_ms
        ]
        cold = [t for t in ttfts([r for r in records if r.derived.get("phase") == "miss"])]
        warm = [t for t in ttfts([r for r in records if r.derived.get("phase") == "hit"])]
        if cold and warm:
            for r in records:
                r.derived["cold_ttft_median_ms"] = statistics.median(cold)
                r.derived["warm_ttft_median_ms"] = statistics.median(warm)
                r.derived["ttft_speedup"] = (
                    statistics.median(cold) / statistics.median(warm)
                    if statistics.median(warm) > 0
                    else None
                )
        return records
