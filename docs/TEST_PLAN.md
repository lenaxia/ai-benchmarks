# Test Plan

How we validate that the harness itself is correct — offline (CI) and against
live endpoints — plus what is deliberately off-the-shelf vs hand-rolled.

## 1. Validation layers

| layer | what it proves | where |
|---|---|---|
| **Unit** | timing math (TTFT/ITL/percentiles/decode+prefill tps), token estimation & filler sizing/uniqueness, hardware parsers (nvidia-smi/rocminfo/lscpu/cpuinfo/meminfo/apple fixtures), config env-expansion + secret redaction, results summarisation | `tests/test_metrics.py`, `test_corpus.py`, `test_hardware.py`, `test_config_results.py` |
| **Wire protocol** | OpenAI-compat SSE client against a stdlib fake server: text streaming, tool-call fragment assembly, usage incl. `cached_tokens`, `[DONE]`, keepalives, 429/5xx retry with backoff, 400 `stream_options` fallback, no-retry on 4xx | `tests/test_provider_openai.py` + `tests/fake_openai.py` |
| **World machinery** | board state machine: claim tokens/leases, heartbeat watermark monotonicity, future-stamp 400, expired-token 409, dependency blocking, evidence gating, criteria-guarded `done`, audit ordering, deterministic ids | `tests/test_board.py` |
| **Sandbox** | path-escape prevention, edit ambiguity errors, bash timeouts + exit codes, PATH shims overriding real CLIs, git works out of the box, glob/grep | `tests/test_env_tools.py` |
| **Loop + sessions** | tool execution ordering/transcript shape, unknown-tool + malformed-args recovery, max_turns/budget stops, subtask delegation with depth limit and metric rollup, board errors surfacing as JSON, fire-and-forget session platform (create/poll/send_message continuation) | `tests/test_agentic_loop.py`, `test_sessions.py` |
| **Workloads** | decode labels/targets, prefill derivation, cache phase machine (miss/warmup/hit), miss prefixes genuinely unique + hit prefixes byte-identical | `tests/test_workloads.py` |
| **Scenario graders** | scripted "perfect agent" walks every scenario to PASS (walk-to-review, reviewer-done, lease stand-down, epic lane, bugfix); scripted lazy/stormy agents FAIL the right checks (no-fix fails, criteria-touching fails, retry-storm fails) | `tests/test_scenarios.py` |
| **CLI** | end-to-end `aibench run` against the fake server produces run.json/perf.jsonl/report.md; `aibench report` regenerates from artifacts; `--only` flag semantics | smoke-tested in dev; keep a manual pass on release |

CI runs layers 1-7 fully offline on 3.11/3.12 (`.github/workflows/ci.yml`).

## 2. Live-endpoint validation (manual, per new server flavor)

Run against every new serving stack before trusting its numbers — server
quirks are the main source of misleading benchmarks:

1. **Wire probe** — streaming + `stream_options.include_usage` accepted? usage
   present? `prompt_tokens_details.cached_tokens` reported? tool calls
   streamed correctly? (both flavors validated: qwen38-flash-next reports
   cached tokens + reasoning; vLLM-backed `default` reports neither reasoning
   nor cached tokens)
2. **Small perf run** (`configs/live-validate.yaml` shape: decode [512, 4k,
   16k], prefill [1k, 8k, 32k], cache 4k) — sanity matrix:
   - TTFT grows monotonically with prompt size
   - prefill tok/s plausible and stable across repeats
   - cache: miss rows show ~0 cached tokens, hit rows show cached ≈ prefix
     size and TTFT drops (validated: 4.3s→1.4s with 3808/4324 cached)
   - **burst-stream detection**: relays that aggregate SSE burst-deliver
     chunks, making inter-token decode timing meaningless — the harness flags
     `burst_stream` per row, nulls net decode tps, and reports gross
     (tokens/total-time) instead. If burst/n > 0, only gross and TTFT (≈e2e)
     are trustworthy through that endpoint; decode speed must be measured
     against the origin server directly.
3. **Agentic smoke** — `plain-bugfix-median` + `board-walk-to-review` must
   PASS for any model you intend to rank; then the full set.

Validated live (2026-10-25, hosted relay):

| model | plain-bugfix | walk-to-review | reviewer-done | epic-lane | lease-recovery | orchestrate | delegate | refactor | feature |
|---|---|---|---|---|---|---|---|---|---|
| qwen38-flash-next | PASS | PASS | - | - | - | - | - | - | - |
| default (qwen38 27b) | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL (contract: made `balance()` a property; grader caught it) |

Agentic loops measured 75-82% prompt-cache utilisation — the per-turn cache
accounting works against a server that reports `cached_tokens`.

## 3. Known caveats

- decode/prefill numbers through aggregating relays are e2e-dominated; the
  burst flag marks them but origin-direct runs are required for decode tps.
- reasoning models: TTFT = first *any* (reasoning-or-content) token — the
  prefill proxy; `ttft_content_ms` records visible-latency separately.
- hosted-endpoint token/s includes provider-side batching noise; rank local
  stacks on perf, hosted models on agentic scores + cost.
- scripted-provider walkthroughs prove grener correctness, not model ability.

## 4. Off-the-shelf vs hand-rolled

**Off the shelf:** httpx (HTTP/SSE), pydantic (config validation), PyYAML,
tiktoken (optional exact prompt sizing), pytest/pytest-asyncio, ruff.

**Deliberately hand-rolled:**

- *SSE parsing instead of the openai SDK* — the SDK hides chunk timing and
  server quirks; we need raw timestamps for TTFT/ITL, `stream_options`
  fallback control, and compatibility with the long tail of OpenAI-ish
  servers (vLLM, llama.cpp, LM Studio, relays).
- *Agentic loop instead of an eval framework* (Inspect, LangSmith, ...) —
  they own the tool loop and world model; our scenarios grade final world
  state (fs + board audit + message records) and depend on a bespoke board
  mock with claim/lease/heartbeat semantics copied from production.
- *Board/session mock* — no off-the-shelf equivalent exists; it is the point.
- *Fake OpenAI server + ScriptedProvider* — tiny (one file each) and exact;
  pulling a mock-server library would not reduce risk.
