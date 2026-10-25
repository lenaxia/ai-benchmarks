from aibench.bench.corpus import TokenEstimator
from aibench.bench.workloads import CacheWorkload, DecodeAtContextWorkload, PrefillWorkload
from aibench.testing import ScriptedProvider, Step


def _usage_provider(n: int):
    return ScriptedProvider(steps=[Step(text="1 2 3", usage={
        "prompt_tokens": 100, "completion_tokens": 3, "total_tokens": 103,
        "prompt_tokens_details": {"cached_tokens": 0},
    }) for _ in range(n)])


async def test_decode_workload_labels_and_rows():
    est = TokenEstimator(use_tiktoken=False)
    wl = DecodeAtContextWorkload([100, 200], output_tokens=64, estimator=est)
    provider = _usage_provider(3)  # 2 levels, no warmup
    rows = await wl.run(provider, repeats=1, warmup=0)
    assert [r.label for r in rows] == ["ctx=100", "ctx=200"]
    assert all(r.response.usage.prompt_tokens == 100 for r in rows)
    assert all(r.derived["decode_tps"] is None or r.derived["decode_tps"] > 0 for r in rows)


async def test_prefill_workload():
    est = TokenEstimator(use_tiktoken=False)
    wl = PrefillWorkload([100], max_tokens=8, estimator=est)
    provider = _usage_provider(1)
    rows = await wl.run(provider, repeats=1, warmup=0)
    assert len(rows) == 1
    assert rows[0].target_output_tokens == 8
    assert "prefill_tps" in rows[0].derived


async def test_cache_workload_phases_and_speedup():
    est = TokenEstimator(use_tiktoken=False)

    class FlakyUsage(ScriptedProvider):
        def __init__(self):
            super().__init__(steps=[])

        async def chat(self, **kw):
            self.i = getattr(self, "i", 0) + 1
            cold = self.i <= 2  # first miss + warmup are cold, then warm
            self._n = 0
            self.steps = [Step(text="ok", usage={
                "prompt_tokens": 120,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 0 if cold else 110},
            })]
            return await super().chat(**kw)

    wl = CacheWorkload(prefix_tokens=60, estimator=est)
    rows = await wl.run(FlakyUsage(), repeats=2, warmup=1)
    phases = [r.derived["phase"] for r in rows]
    # warmup=1 + 2 misses, then 1 hit_warmup + 2 hits
    assert phases == ["miss", "miss", "miss", "hit_warmup", "hit", "hit"]
    hit_rows = [r for r in rows if r.derived["phase"] == "hit"]
    assert all(r.response.usage.cached_tokens == 110 for r in hit_rows)
    assert all("warm_ttft_median_ms" in r.derived for r in rows)


async def test_cache_miss_prefixes_unique():
    est = TokenEstimator(use_tiktoken=False)
    wl = CacheWorkload(prefix_tokens=50, estimator=est)
    captured = []

    class Capture(ScriptedProvider):
        async def chat(self, *, messages, **kw):
            captured.append(messages[0]["content"])
            return await super().chat(messages=messages, **kw)

    provider = Capture(steps=[])
    for _ in range(6):
        provider.steps.append(Step(text="ok"))
    await wl.run(provider, repeats=2, warmup=1)
    # the three miss prompts must differ in their prefix
    misses = captured[:3]
    assert len({m[:80] for m in misses}) == 3
    # the three hit prompts must share their prefix
    hits = captured[3:]
    assert len({h[:80] for h in hits}) == 1
