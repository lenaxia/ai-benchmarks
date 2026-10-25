import os

from aibench.config import RunConfig
from aibench.results.writer import (
    ResultsWriter,
    build_report,
    summarize_agentic,
    summarize_perf,
)

CONFIG_YAML = """
run_name: test-run
provider:
  type: openai
  base_url: http://localhost:9999/v1
  model: test-model
  api_key: ${TEST_KEY_XYZ}
perf:
  repeats: 2
  decode:
    context_levels: [100, 200]
agentic:
  scenarios: [plain-bugfix-median]
"""


def test_config_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY_XYZ", "sekret")
    p = tmp_path / "cfg.yaml"
    p.write_text(CONFIG_YAML)
    cfg = RunConfig.from_yaml(str(p))
    assert cfg.provider.api_key == "sekret"
    assert cfg.perf.repeats == 2
    assert cfg.perf.decode.context_levels == [100, 200]
    assert cfg.agentic.scenarios == ["plain-bugfix-median"]
    # secrets redacted in public dict
    assert cfg.to_public_dict()["provider"]["api_key"] == "***redacted***"


def test_config_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("run_name: x\n")
    cfg = RunConfig.from_yaml(str(p))
    assert cfg.provider.type == "openai"
    assert cfg.perf.cache.enabled is True
    assert cfg.agentic.subtask.max_turns == 12


def test_missing_env_becomes_empty(tmp_path):
    os.environ.pop("TEST_KEY_XYZ", None)
    p = tmp_path / "cfg.yaml"
    p.write_text("provider: {api_key: '${TEST_KEY_XYZ}'}\n")
    cfg = RunConfig.from_yaml(str(p))
    assert cfg.provider.api_key == ""


def test_results_writer_artifacts(tmp_path):
    writer = ResultsWriter(str(tmp_path), "my run")
    writer.write_run_meta({"run_name": "my run"}, {"cpu": {}})
    writer.write_perf_rows([
        {"workload": "decode", "label": "ctx=100", "decode_tps": 42.0, "error": None,
         "timing": {"ttft_ms": 100.0}},
        {"workload": "decode", "label": "ctx=100", "decode_tps": 48.0, "error": None,
         "timing": {"ttft_ms": 120.0}},
    ])
    assert (writer.dir / "run.json").exists()
    assert (writer.dir / "perf.jsonl").exists()
    perf = summarize_perf([
        {"workload": "decode", "label": "ctx=100", "decode_tps": 42.0, "error": None,
         "timing": {"ttft_ms": 100.0}},
        {"workload": "decode", "label": "ctx=100", "decode_tps": 48.0, "error": None,
         "timing": {"ttft_ms": 120.0}},
    ])
    assert perf["decode"]["ctx=100"]["decode_tps_median"] == 45.0
    report = build_report("r", "hw", "m", perf, {})
    assert "ctx=100" in report and "decode tok/s" in report
    writer.write_report(report)
    assert (writer.dir / "report.md").exists()


def test_summarize_agentic():
    from aibench.agentic.runner import ScenarioResult

    r1 = ScenarioResult(scenario="s", passed=True, score=1.0)
    r1.metrics = {"turns": 4, "tool_calls": 10, "prompt_tokens": 1000, "cached_tokens": 600}
    r2 = ScenarioResult(scenario="s", passed=False, score=0.5)
    r2.metrics = {"turns": 6, "tool_calls": 20, "prompt_tokens": 1000, "cached_tokens": 200}
    s = summarize_agentic([r1, r2])
    assert s["s"]["attempts"] == 2
    assert s["s"]["passed"] == 1
    assert s["s"]["turns_avg"] == 5.0
    assert s["s"]["cached_tokens_total"] == 800


def test_scenario_names_unique_and_listable():
    from aibench.agentic.scenarios import list_scenarios

    rows = list_scenarios()
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names))
    assert any(r["toolset"] == "plain" for r in rows)
    assert any(r["toolset"] == "safespace" for r in rows)
