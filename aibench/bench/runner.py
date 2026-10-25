"""Perf suite runner: assemble workloads from config, execute, collect rows."""

from __future__ import annotations

from typing import Any

from aibench.bench.corpus import TokenEstimator
from aibench.bench.workloads import (
    CacheWorkload,
    DecodeAtContextWorkload,
    PrefillWorkload,
)
from aibench.config import RunConfig
from aibench.providers.base import Provider
from aibench.results.writer import ResultsWriter


async def run_perf_suite(
    cfg: RunConfig, provider: Provider, writer: ResultsWriter
) -> list[dict[str, Any]]:
    perf = cfg.perf
    estimator = TokenEstimator(chars_per_token=perf.chars_per_token)
    workloads = []
    if perf.decode.enabled:
        workloads.append(
            DecodeAtContextWorkload(
                perf.decode.context_levels,
                output_tokens=perf.decode.output_tokens,
                estimator=estimator,
                temperature=perf.temperature,
            )
        )
    if perf.prefill.enabled:
        workloads.append(
            PrefillWorkload(
                perf.prefill.prompt_levels,
                max_tokens=perf.prefill.max_tokens,
                estimator=estimator,
                temperature=perf.temperature,
            )
        )
    if perf.cache.enabled:
        workloads.append(
            CacheWorkload(
                prefix_tokens=perf.cache.prefix_tokens,
                estimator=estimator,
                temperature=perf.temperature,
            )
        )

    all_rows: list[dict[str, Any]] = []
    for wl in workloads:
        repeats = (
            perf.cache.repeats if isinstance(wl, CacheWorkload) else perf.repeats
        )
        warmup = (
            perf.cache.warmup if isinstance(wl, CacheWorkload) else perf.warmup
        )
        records = await wl.run(provider, repeats, warmup)
        rows = [r.to_row() for r in records]
        writer.write_perf_rows(rows)
        all_rows.extend(rows)
        ok = sum(1 for r in rows if not r.get("error"))
        print(f"  [perf:{wl.name}] {ok}/{len(rows)} requests ok")
    return all_rows


async def run_agentic_suite(
    cfg: RunConfig, provider: Provider, writer: ResultsWriter
) -> list[Any]:
    from aibench.agentic.runner import run_scenario
    from aibench.agentic.scenarios import list_scenarios

    wanted = cfg.agentic.scenarios
    names = (
        [s["name"] for s in list_scenarios()]
        if "all" in wanted or not wanted
        else list(wanted)
    )
    results = []
    for repeat in range(cfg.agentic.repeats):
        for i, name in enumerate(names, 1):
            from aibench.agentic.scenarios import get_scenario

            print(f"  [agentic {i}/{len(names)}] {name} (repeat {repeat + 1})")
            result = await run_scenario(
                get_scenario(name),
                provider,
                subagent_max_turns=cfg.agentic.subtask.max_turns,
                max_depth=cfg.agentic.subtask.max_depth,
            )
            writer.write_scenario(result, repeat + 1)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            failed = [c.name for c in result.checks if not c.passed]
            print(
                f"      {status} score={result.score:.2f} turns={result.metrics.get('turns')} "
                f"tools={result.metrics.get('tool_calls')}"
                + (f" failed={failed}" if failed else "")
            )
    return results
