"""Decode throughput at context levels.

For each context level the conversation is padded with filler to ~N prompt
tokens, then the model generates ``output_tokens`` tokens. Decode tok/s is
computed from the inter-token window so prefill cost does not pollute it.
Filler is unique per request so TTFT stays a cold-prefill measurement too.
"""

from __future__ import annotations

from aibench.bench.corpus import DECODE_INSTRUCTION, TokenEstimator
from aibench.bench.workloads.base import RequestRecord, Workload
from aibench.metrics import decode_tps, decode_tps_gross, is_burst_stream, prefill_tps
from aibench.providers.base import Provider


class DecodeAtContextWorkload(Workload):
    name = "decode"

    def __init__(
        self,
        context_levels: list[int],
        output_tokens: int = 256,
        estimator: TokenEstimator | None = None,
        temperature: float | None = None,
    ):
        self.context_levels = sorted(context_levels)
        self.output_tokens = output_tokens
        self.estimator = estimator or TokenEstimator()
        self.temperature = temperature

    async def _one(self, provider: Provider, level: int, seq: int) -> RequestRecord:
        prompt = self.estimator.context_prompt(
            level, instruction=DECODE_INSTRUCTION, seed=1000 + level, unique=True
        )
        resp = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.output_tokens,
            temperature=self.temperature,
        )
        out_tok = resp.output_tokens()
        rec = RequestRecord(
            workload=self.name,
            label=f"ctx={level}",
            target_prompt_tokens=level,
            target_output_tokens=self.output_tokens,
            response=resp,
        )
        rec.derived["decode_tps"] = None if is_burst_stream(resp.timing) else decode_tps(
            resp.timing, out_tok
        )
        rec.derived["decode_tps_gross"] = decode_tps_gross(resp.timing, out_tok)
        rec.derived["burst_stream"] = is_burst_stream(resp.timing)
        if resp.prompt_tokens():
            rec.derived["prefill_tps"] = prefill_tps(
                resp.timing, resp.prompt_tokens(), resp.usage.cached_tokens if resp.usage else None
            )
        return rec

    async def run(self, provider: Provider, repeats: int, warmup: int) -> list[RequestRecord]:
        records: list[RequestRecord] = []
        first = self.context_levels[0] if self.context_levels else 512
        for _ in range(warmup):
            await self._one(provider, first, -1)
        for level in self.context_levels:
            for rep in range(repeats):
                records.append(await self._one(provider, level, rep))
        return records
