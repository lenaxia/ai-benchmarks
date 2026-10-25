# ai-benchmarks

A benchmarking harness for LLM inference — models **and** the hardware they run
on — covering raw serving performance and agentic tool-loop competence.

Targets OpenAI-compatible endpoints first (vLLM, llama.cpp server, LM Studio,
TGI, SGLang, OpenRouter, OpenAI, ...) with a provider abstraction for other
wire protocols later.

Three measurement families:

| family | what it measures |
|---|---|
| **Performance** | decode tok/s at context levels, prefill speed (TTFT), prefix-cache hit/miss behaviour with deliberately varied contexts |
| **Hardware** | GPU(s) / CPU / RAM snapshot captured with every run, manual overrides for remote endpoints |
| **Agentic** | tool-loop scenarios: plain coding tools (read/write/edit/bash/subtask) and a mocked **llmsafespaces** board/session/task platform with claims, leases, heartbeats, stage guards and criteria |

## Install

```bash
git clone https://github.com/lenaxia/ai-benchmarks
cd ai-benchmarks
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Optional: `pip install -e ".[tok]"` enables exact tiktoken counts for prompt
sizing (heuristic chars/token is used otherwise; server-reported usage is
always preferred).

## Quickstart

```bash
# what would we be running on?
aibench hardware

# list the agentic scenarios
aibench scenarios

# run everything against a local server
cp configs/local-vllm.yaml myrun.yaml   # edit base_url/model
aibench run myrun.yaml

# agentic only, single scenario, extra repeats
aibench run myrun.yaml --only agentic --scenario board-epic-lane --repeat 3

# regenerate report.md from an existing results dir
aibench report results/myrun-20260101-120000
```

Results land in `results/<run_name>-<timestamp>/`:

```
run.json        config echo + hardware snapshot (secrets redacted)
perf.jsonl      one row per request (timing, usage, derived metrics)
agentic.jsonl   one row per scenario attempt (checks, metrics, board state)
transcripts/    full message logs per scenario attempt
summary.json    aggregated medians + scoreboard
report.md       human-readable tables
```

## Configuration

YAML, with `${ENV_VAR}` expansion (never commit keys):

```yaml
run_name: rtx4090-qwen32b
provider:
  type: openai              # provider registry name; openai = OpenAI-compatible
  base_url: http://localhost:8000/v1
  model: Qwen2.5-32B-Instruct
  api_key: ${LOCAL_API_KEY}
  timeout_s: 600
  max_retries: 3
  extra_headers: {}
hardware:
  capture: true             # nvidia-smi / rocminfo / lscpu / meminfo auto-detect
  manual:                   # wins over detection — describe the serving box
    gpus: [{vendor: nvidia, name: RTX 4090, vram_gb: 24}]
    cpu: {model: Ryzen 9 7950X, physical_cores: 16, threads: 32}
    ram_gb: 64
perf:
  repeats: 3
  warmup: 1
  decode:  {context_levels: [512, 2048, 8192, 32768, 131072], output_tokens: 256}
  prefill: {prompt_levels: [1024, 8192, 32768, 131072], max_tokens: 32}
  cache:   {prefix_tokens: 4096, repeats: 5, warmup: 1}
agentic:
  scenarios: [all]          # or a list of scenario names
  repeats: 1
  subtask: {max_depth: 1, max_turns: 12}
```

See `configs/local-vllm.yaml` and `configs/remote-api.yaml` for full examples.

## Metrics

Per request (streamed; from wall-clock chunk timestamps + server usage):

| metric | definition |
|---|---|
| `ttft_ms` | request-sent → first output token |
| `decode_tps` | (output_tokens − 1) / (last − first token time) — prefill excluded |
| `prefill_tps` | non-cached prompt tokens / TTFT |
| `itl_*_ms` | inter-token latency mean/p50/p95/max |
| `cached_tokens` | server-reported prefix-cache hits (`prompt_tokens_details.cached_tokens`) |

Decode uses the inter-token window so a slow prefill never pollutes decode
throughput; prefill uses TTFT so decode never pollutes prefill.

Agentic loops additionally record, **per turn**: prompt/completion/cached
tokens. Agent loops re-send the whole conversation every turn, so prefix-cache
utilisation is a first-class result — `report.md` shows cache use per scenario.

## Perf workloads

- **decode** — pads the conversation to each context level with random-word
  filler (unique per request, so TTFT stays cold), then generates
  `output_tokens`. Reports decode tok/s per context level.
- **prefill** — long prompts, ~1-sentence replies. TTFT ≈ prefill time;
  `prefill_tps = non-cached prompt tokens / TTFT`.
- **cache** — miss phase sends UUID-salted unique prefixes (a prefix-caching
  server must report ~0 cached tokens, cold TTFT); hit phase repeats one
  byte-identical prefix with a tiny varying suffix (cached tokens ≈ prefix,
  TTFT drops). Reports both timing-based speedup and server-reported cache
  tokens, since not every server reports the field.

## Agentic harness

Each scenario runs the benchmarked model through a real tool loop in a
sandboxed temp workspace (path-confined file ops, bash with timeouts+output
truncation, git pre-configured) and grades deterministic checks against the
final world state — filesystem, board state, audit trail ordering, message
records and budgets.

### Toolsets

- **plain** — `read`, `write`, `edit` (exact-match replace), `bash`, `glob`,
  `grep`, `subtask` (blocking read-only delegation with depth limit)
- **safespace** — plain tools **plus** the mocked llmsafespaces platform:
  - board: `board_create_project/epic/task`, `board_claim`, `board_read`,
    `board_heartbeat`, `board_move_stage`, `board_update_task`, `board_list`
  - sessions: `session_create` (fire-and-forget **real nested agent loop**),
    `session_list/read/metadata`, `send_message`
  - `get_datetime`, `subtask`

### Board semantics (mocked faithfully from production behaviour)

- claims mint a `claim_token` + 15-min lease; in-flight writes need the token
- `board_heartbeat` requires a real current timestamp strictly newer than the
  last (future stamps → `400 future_stamp`; stale → `409 not_progress`)
- `in_review` needs the token + an evidenceUrl
- `done` is refused (`409 criteria_outstanding`) until every acceptance
  criterion is marked met — reproduce the guard-error-driven flow agents
  actually hit
- a stale/expired token yields `409 claim_token` — the stand-down signal
- every mutation appends to an audit log; checks assert ordering (e.g.
  "A completed before B was claimed", "evidence read before criteria flagged")

### Scenarios

| name | toolset | archetype |
|---|---|---|
| `plain-bugfix-median` | plain | red-first bugfix (failing test → fix → green, no test weakening) |
| `plain-feature-ledger` | plain | feature + tests + clean git commit |
| `plain-refactor-rename` | plain | mechanical rename across call sites |
| `plain-delegate-router` | plain | mandatory subtask delegation, then fix |
| `board-walk-to-review` | safespace | worker lane: claim → heartbeat → evidence → in_review, criteria left to reviewer |
| `board-reviewer-done` | safespace | reviewer: verify evidence → setCriteriaMet → done → verdict message |
| `board-epic-lane` | safespace | manager: build epic, dependency-ordered tasks worked to done |
| `board-lease-recovery` | safespace | lapsed lease: 409 claim_token → stand down or re-claim, no retry storm |
| `board-session-orchestrate` | safespace | spawn worker session, poll, verify artifact, thank, report |

Score = fraction of checks passed; `passed` requires all of them (budget and
wall-time checks are always appended).

## Extending

**Providers** — implement `aibench.providers.base.Provider` (one async
`chat()` returning a normalised `ChatResponse` with stream timing) and
register it:

```python
from aibench.providers.base import Provider, register_provider

@register_provider("anthropic")
class AnthropicProvider(Provider): ...
```

then set `provider.type: anthropic` in the config.

**Tools** — subclass `aibench.agentic.tools.base.Tool` (name, description,
JSON-schema parameters, async `run(args, ctx)`), add it in
`aibench/agentic/registry.py`.

**Scenarios** — build a `Scenario` (setup seeds the world, task is a charter
template, checks are pure functions of final state) and append it in
`aibench/agentic/scenarios/`:

```python
my_scenario = Scenario(
    name="plain-mine",
    title="...",
    toolset="plain",
    task="Fix the bug in {module}",      # {placeholders} filled from setup data
    setup=lambda bundle: bundle.env.write_file("module", "..."),
    checks=[("fixed", lambda ctx: ("ok" in ctx.env.read_file("module"), "detail"))],
)
```

Validate custom scenarios offline without a server using
`aibench.testing.ScriptedProvider` (pre-scripted assistant turns — the same
trick the test suite uses to assert graders catch both success and failure).

## Offline testing / CI

```bash
ruff check aibench tests
pytest -q
```

The suite runs fully offline: a stdlib fake OpenAI SSE server covers the wire
protocol (streaming, tool-call fragments, usage details, 4xx/5xx retries,
stream_options fallback) and `ScriptedProvider` drives deterministic
scenario walkthroughs.

## Architecture

```
aibench/
├── providers/      # Provider ABC + OpenAI-compatible SSE client (httpx)
├── hardware/       # GPU/CPU/RAM detection + manual overrides
├── metrics/        # StreamTracker: TTFT / ITL / decode+prefill throughput
├── bench/          # corpus (token sizing, cache-defeating filler) + workloads
├── agentic/
│   ├── env.py      # sandboxed workspace: fs ops, bash, PATH shims, call log
│   ├── board.py    # mocked task board: claims, leases, heartbeats, stage guards
│   ├── sessions.py # mocked session platform driving real nested loops
│   ├── loop.py     # tool loop with budgets + per-turn metrics
│   ├── registry.py # toolset assembly (plain / safespace / readonly)
│   ├── runner.py   # scenario execution + grading
│   └── scenarios/  # the scenario library
├── results/        # JSONL writers + markdown report builder
├── testing.py      # ScriptedProvider for offline runs
└── cli.py          # aibench hardware | scenarios | run | report
```

## License

MIT
