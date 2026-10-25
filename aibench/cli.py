"""aibench CLI.

    aibench hardware [--json]          show detected GPU/CPU/RAM
    aibench scenarios                  list agentic scenarios
    aibench run CONFIG.yaml [flags]    run a benchmark suite
    aibench report RESULTS_DIR         rebuild report.md from artifacts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from aibench import __version__
from aibench.agentic.scenarios import list_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aibench", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    hw = sub.add_parser("hardware", help="detect and print hardware")
    hw.add_argument("--json", action="store_true")
    hw.add_argument("--manual", help="YAML/JSON file with manual hardware overrides")

    sub.add_parser("scenarios", help="list agentic scenarios")

    run = sub.add_parser("run", help="run a benchmark suite from a config file")
    run.add_argument("config", help="path to run config YAML")
    run.add_argument("--only", choices=["perf", "agentic", "all"], default="all")
    run.add_argument("--scenario", action="append", help="restrict agentic scenarios (repeatable)")
    run.add_argument("--repeat", type=int, help="override agentic repeats")
    run.add_argument("--output-dir", help="override output directory")

    rep = sub.add_parser("report", help="rebuild report.md from a results directory")
    rep.add_argument("results_dir")

    args = parser.parse_args(argv)

    if args.cmd == "hardware":
        return _cmd_hardware(args)
    if args.cmd == "scenarios":
        return _cmd_scenarios()
    if args.cmd == "run":
        return asyncio.run(_cmd_run(args))
    if args.cmd == "report":
        return _cmd_report(args)
    return 2


def _cmd_hardware(args) -> int:
    from aibench.hardware.detect import detect_hardware_sync

    manual = None
    if args.manual:
        import yaml

        with open(args.manual) as f:
            manual = yaml.safe_load(f) or None
    snap = detect_hardware_sync(manual)
    if args.json:
        print(json.dumps(snap.to_dict(), indent=2))
    else:
        print(snap.short())
        print(json.dumps(snap.to_dict(), indent=2))
    return 0


def _cmd_scenarios() -> int:
    rows = list_scenarios()
    print(f"{'name':34} {'toolset':9} checks  turns  title")
    for r in rows:
        print(
            f"{r['name']:34} {r['toolset']:9} {len(r['checks']):>3}    "
            f"{r['max_turns']:>3}    {r['title']}"
        )
    return 0


async def _cmd_run(args) -> int:
    from aibench.bench.runner import run_agentic_suite, run_perf_suite
    from aibench.config import RunConfig
    from aibench.hardware.detect import detect_hardware
    from aibench.providers.base import create_provider
    from aibench.results.writer import (
        ResultsWriter,
        build_report,
        summarize_agentic,
        summarize_perf,
    )

    cfg = RunConfig.from_yaml(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.scenario:
        cfg.agentic.scenarios = args.scenario
    if args.repeat is not None:
        cfg.agentic.repeats = args.repeat
    if args.only == "perf":
        cfg.agentic.enabled = False
    if args.only == "agentic":
        cfg.perf.enabled = False

    print(f"aibench run: {cfg.run_name}")
    hardware = None
    if cfg.hardware.capture:
        hardware = await detect_hardware(cfg.hardware.manual)
        print(f"hardware: {hardware.short()}")

    provider = create_provider(
        cfg.provider.type,
        base_url=cfg.provider.base_url,
        model=cfg.provider.model,
        api_key=cfg.provider.api_key,
        timeout_s=cfg.provider.timeout_s,
        max_retries=cfg.provider.max_retries,
        extra_headers=cfg.provider.extra_headers,
        stream_options=cfg.provider.stream_options,
        request_delay_s=cfg.provider.request_delay_s,
    )
    writer = ResultsWriter(cfg.output_dir, cfg.run_name)
    writer.write_run_meta(cfg.to_public_dict(), hardware.to_dict() if hardware else {})
    print(f"results: {writer.dir}")

    perf_summary: dict = {}
    agentic_results: list = []
    try:
        if cfg.perf.enabled:
            print("== performance ==")
            rows = await run_perf_suite(cfg, provider, writer)
            perf_summary = summarize_perf(rows)
        if cfg.agentic.enabled:
            print("== agentic ==")
            agentic_results = await run_agentic_suite(cfg, provider, writer)
    finally:
        await provider.close()

    agentic_summary = summarize_agentic(agentic_results)
    summary = {"run_name": cfg.run_name, "perf": perf_summary, "agentic": agentic_summary}
    writer.write_summary(summary)
    report = build_report(
        cfg.run_name,
        hardware.short() if hardware else "n/a",
        cfg.provider.model,
        perf_summary,
        agentic_summary,
    )
    writer.write_report(report)
    print(f"\ndone -> {writer.dir}")
    print(f"report -> {writer.dir / 'report.md'}")
    return 0


def _cmd_report(args) -> int:
    from aibench.results.writer import (
        build_report,
        summarize_agentic,
        summarize_perf,
    )

    d = Path(args.results_dir)
    rows = []
    perf_path = d / "perf.jsonl"
    if perf_path.exists():
        rows = [json.loads(line) for line in perf_path.read_text().splitlines() if line]
    agentic_results = []
    agentic_path = d / "agentic.jsonl"
    if agentic_path.exists():
        from aibench.agentic.runner import ScenarioResult

        for line in agentic_path.read_text().splitlines():
            if line:
                data = json.loads(line)
                r = ScenarioResult(scenario=data["scenario"])
                r.score = data["score"]
                r.passed = data["passed"]
                r.metrics = data["metrics"]
                agentic_results.append(r)
    meta = {}
    meta_path = d / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    report = build_report(
        meta.get("config", {}).get("run_name", d.name),
        "n/a",
        meta.get("config", {}).get("provider", {}).get("model", "?"),
        summarize_perf(rows),
        summarize_agentic(agentic_results),
    )
    out = d / "report.md"
    out.write_text(report)
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
