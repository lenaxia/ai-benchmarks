"""Prefill / TTFT workload: long prompts, minimal generation.

Each request sends a ~N-token prompt with a one-sentence reply demanded,
so TTFT is dominated by prefill. prefill_tps = non-cached prompt tokens /
TTFT. Unique filler per request keeps every measurement cold.
"""

from __future__ import annotations

from aibench.bench.corpus import SUMMARIZE_INSTRUCTION, TokenEstimator
from aibench.bench.workloads.base import RequestRecord, Workload
from aibench.metrics import prefill_tps
from aibench.providers.base import Provider


class PrefillWorkload(Workload):
    name = "prefill"

    def __init__(
        self,
        prompt_levels: list[int],
        max_tokens: int = 32,
        estimator: TokenEstimator | None = None,
        temperature: float | None = None,
    ):
        self.prompt_levels = sorted(prompt_levels)
        self.max_tokens = max_tokens
        self.estimator = estimator or TokenEstimator()
        self.temperature = temperature

    async def _one(self, provider: Provider, level: int) -> RequestRecord:
        prompt = self.estimator.context_prompt(
            level, instruction=SUMMARIZE_INSTRUCTION, seed=2000 + level, unique=True
        )
        resp = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        rec = RequestRecord(
            workload=self.name,
            label=f"prompt={level}",
            target_prompt_tokens=level,
            target_output_tokens=self.max_tokens,
            response=resp,
        )
        rec.derived["prefill_tps"] = prefill_tps(
            resp.timing,
            resp.prompt_tokens(),
            resp.usage.cached_tokens if resp.usage else None,
        )
        return rec

    async def run(self, provider: Provider, repeats: int, warmup: int) -> list[RequestRecord]:
        records: list[RequestRecord] = []
        first = self.prompt_levels[0] if self.prompt_levels else 1024
        for _ in range(warmup):
            await self._one(provider, first)
        for level in self.prompt_levels:
            for _ in range(repeats):
                records.append(await self._one(provider, level))
        return records
