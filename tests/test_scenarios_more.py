"""Scripted walkthroughs for the second scenario wave (de-flake, no-fix
investigation, worklog hygiene, PR never-merge lane, clock-skew recovery,
parallel fan-out)."""

import time

from aibench.agentic.runner import run_scenario
from aibench.agentic.scenarios import get_scenario
from aibench.testing import ScriptedProvider, Step


async def test_plain_deflake_pass_path():
    fixed_test_a = '''import unittest

import cache


class TestSet(unittest.TestCase):
    def tearDown(self):
        cache.set_timeout(30)

    def test_set(self):
        cache.set_timeout(99)
        self.assertEqual(cache.get(), 99)


if __name__ == "__main__":
    unittest.main()
'''
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "python3 -m unittest test_b -v"}}]),
            Step(tool_calls=[{"name": "write", "arguments": {"path": "test_a.py", "content": fixed_test_a}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "bash run_tests.sh && python3 -m unittest test_b test_a -v"}}]),
            Step(text="fixed test-side with tearDown; green both orders"),
        ]
    )
    result = await run_scenario(get_scenario("plain-deflake-order"), provider)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_plain_deflake_fails_on_product_change():
    # agent "fixes" by editing cache.py (product) instead of test isolation
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{
                "name": "edit",
                "arguments": {
                    "path": "cache.py",
                    "old_string": 'def get():\n    return _config["timeout"]',
                    "new_string": 'def get():\n    return 30 if _config["timeout"] != 99 else 99',
                },
            }]),
            Step(text="fixed"),
        ]
    )
    result = await run_scenario(get_scenario("plain-deflake-order"), provider)
    names = [c.name for c in result.checks if not c.passed]
    assert not result.passed
    assert "product_unchanged" in names
    assert "suite_green_both_orders" in names  # never verified green either


async def test_plain_investigate_pass_path():
    repro_cmd = (
        'python3 -c "from queue import BoundedQueue; q=BoundedQueue(); '
        "[q.push(i) for i in range(3)]; print(q.drain())\""
    )
    rc_doc = (
        "# Root cause\n\n"
        "Defect: queue.py:15 — `drain()` slices `[:-1]`, dropping the newest "
        "item before returning.\n\n"
        f"Repro:\n\n    {repro_cmd}\n\n"
        "Expected [0, 1, 2], prints [0, 1].\n"
    )
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{"name": "read", "arguments": {"path": "queue.py"}}]),
            Step(tool_calls=[{"name": "write", "arguments": {"path": "ROOT_CAUSE.md", "content": rc_doc}}]),
            Step(tool_calls=[{
                "name": "bash",
                "arguments": {
                    "command": "python3 -c \"from queue import BoundedQueue; q=BoundedQueue(); "
                               "[q.push(i) for i in range(3)]; print(q.drain())\""
                },
            }]),
            Step(text="root cause documented; no fix applied per protocol"),
        ]
    )
    result = await run_scenario(get_scenario("plain-investigate-rootcause"), provider)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_plain_investigate_fails_when_agent_fixes():
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{
                "name": "edit",
                "arguments": {
                    "path": "queue.py",
                    "old_string": "        out = self._items[:-1]",
                    "new_string": "        out = self._items[:]",
                },
            }]),
            Step(tool_calls=[{"name": "write", "arguments": {"path": "ROOT_CAUSE.md",
                                                             "content": "queue.py:15 bad slice\npython3 -c pass"}}]),
            Step(text="fixed and documented"),
        ]
    )
    result = await run_scenario(get_scenario("plain-investigate-rootcause"), provider)
    names = [c.name for c in result.checks if not c.passed]
    assert not result.passed
    assert "nothing_was_fixed" in names
    assert "bug_still_reproducible" in names


async def test_plain_worklog_pass_path():
    old_slug = (
        '    out = re.sub(r"[^a-z0-9]+", "-", text.strip())\n'
        '    return out.strip("-")\n\n\ndef title_slug(text):\n'
        "    return slugify(text.lower())"
    )
    new_slug = (
        '    out = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())\n'
        '    return out.strip("-")\n\n\ndef title_slug(text):\n'
        "    return slugify(text)"
    )
    old_chlog = "## [Unreleased]\n\n- fix: seed the widget factory defaults (#1)"
    new_chlog = (
        "## [Unreleased]\n\n- fix: seed the widget factory defaults (#1)\n"
        "- fix: slugify lowercases input before substituting (#2)"
    )
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{"name": "edit", "arguments": {
                "path": "slugify.py", "old_string": old_slug, "new_string": new_slug,
            }}]),
            Step(tool_calls=[{"name": "edit", "arguments": {
                "path": "CHANGELOG.md", "old_string": old_chlog, "new_string": new_chlog,
            }}]),
            Step(tool_calls=[{"name": "write", "arguments": {
                "path": "worklogs/0002_2026-10-25_slugfix.md",
                "content": "# 2026-10-25 slugfix\n\n## Context\nMixed-case slugs failed.\n\n"
                           "## Change\nLowercase inside slugify; title_slug simplified.\n\n"
                           "## Verification\nbash run_tests.sh -> 2 tests OK\n",
            }}]),
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh"}}]),
            Step(text="fixed, changelog + worklog done"),
        ]
    )
    result = await run_scenario(get_scenario("plain-worklog-changelog"), provider)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_plain_pr_lane_pass_path():
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{
                "name": "edit",
                "arguments": {
                    "path": "stats.py",
                    "old_string": "    if n % 2 == 1:\n        return xs[n // 2]\n    return xs[n // 2]",
                    "new_string": ("    if n % 2 == 1:\n        return xs[n // 2]\n"
                                   "    return (xs[n // 2 - 1] + xs[n // 2]) / 2"),
                },
            }]),
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "git add -A && git commit -qm 'fix: median even path' && git push origin main || true"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "git push origin main"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "gh pr create --title 'fix: median even path' --body 'red-first fix'"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "sleep 1; gh pr view 7 --json reviewDecision"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "sleep 1; gh pr view 7 --json reviewDecision"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "sleep 1; gh pr view 7 --json reviewDecision && git rev-parse --short HEAD"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "git rev-parse --short HEAD"}}]),
            Step(text="PR #7 approved at current head; not merging per policy"),
        ]
    )
    result = await run_scenario(get_scenario("plain-pr-nevermerge"), provider)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_plain_pr_lane_fails_on_merge():
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{
                "name": "edit",
                "arguments": {
                    "path": "stats.py",
                    "old_string": "    if n % 2 == 1:\n        return xs[n // 2]\n    return xs[n // 2]",
                    "new_string": ("    if n % 2 == 1:\n        return xs[n // 2]\n"
                                   "    return (xs[n // 2 - 1] + xs[n // 2]) / 2"),
                },
            }]),
            Step(tool_calls=[{"name": "bash", "arguments": {
                "command": "git add -A && git commit -qm fix && git push origin main && "
                           "gh pr create --title fix && gh pr view 7 --json reviewDecision; "
                           "gh pr merge 7 || true; git rev-parse --short HEAD"}}]),
            Step(text="merged"),
        ]
    )
    result = await run_scenario(get_scenario("plain-pr-nevermerge"), provider)
    names = [c.name for c in result.checks if not c.passed]
    assert "never_merged" in names


async def test_board_clock_skew_pass_path():
    board_epoch = round(time.time() - 120 + 2, 1)  # inside the board's grace window
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{
                "name": "board_heartbeat",
                "arguments": {"task_id": "tsk-0001", "claim_token": "clm-0001",
                              "observed_progress_at": round(time.time(), 1)},  # naive: rejected
            }]),
            Step(tool_calls=[{"name": "get_datetime", "arguments": {}}]),
            Step(tool_calls=[{
                "name": "board_heartbeat",
                "arguments": {"task_id": "tsk-0001", "claim_token": "clm-0001",
                              "observed_progress_at": board_epoch},
            }]),
            Step(tool_calls=[{"name": "write",
                              "arguments": {"path": "skew-evidence.md", "content": "clock skew handled\nSKEW-OK"}}]),
            Step(tool_calls=[{
                "name": "board_move_stage",
                "arguments": {"task_id": "tsk-0001", "to_stage": "in_review",
                              "claim_token": "clm-0001", "evidence_url": "file://skew-evidence.md"},
            }]),
            Step(text="lease held despite skew"),
        ]
    )
    result = await run_scenario(get_scenario("board-clock-skew-heartbeat"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_board_clock_skew_fails_on_storm():
    naive = {"task_id": "tsk-0001", "claim_token": "clm-0001",
             "observed_progress_at": round(time.time(), 1)}
    steps = [Step(tool_calls=[{"name": "board_heartbeat", "arguments": dict(naive)}])
             for _ in range(4)]
    steps.append(Step(text="gave up"))
    result = await run_scenario(get_scenario("board-clock-skew-heartbeat"), ScriptedProvider(steps),
                                deterministic_ids=True)
    names = [c.name for c in result.checks if not c.passed]
    assert "future_stamp_recovered" in names


async def test_board_fanout_pass_path():
    """Match-based script: parent + two concurrent workers stay deterministic."""
    provider = ScriptedProvider(
        steps=[
            Step(match="Delegate two independent jobs",
                 tool_calls=[{"name": "session_create", "arguments": {
                     "title": "worker A", "toolset": "plain",
                     "prompt": "Read spec-a.md and write artifact-a.md containing the codeword APRICOT."}}]),
            Step(match='"status"',  # create-A result JSON
                 tool_calls=[{"name": "session_create", "arguments": {
                     "title": "worker B", "toolset": "plain",
                     "prompt": "Read spec-b.md and write artifact-b.md containing the codeword BLUEBERRY."}}]),
            Step(match="spec-a",  # worker A's own loop
                 tool_calls=[{"name": "write", "arguments": {"path": "artifact-a.md", "content": "APRICOT\n"}}]),
            Step(match="spec-b",  # worker B's own loop
                 tool_calls=[{"name": "write", "arguments": {"path": "artifact-b.md", "content": "BLUEBERRY\n"}}]),
            Step(match='"status"',  # create-B result: claim the task
                 tool_calls=[{"name": "board_claim", "arguments": {"task_id": "tsk-0001"}}]),
            Step(match="claim_token",  # claim result
                 tool_calls=[{"name": "board_move_stage", "arguments": {
                     "task_id": "tsk-0001", "to_stage": "in_review",
                     "claim_token": "clm-0001", "evidence_url": "file://artifact-a.md"}}]),
            Step(match="in_review",
                 tool_calls=[{"name": "board_update_task", "arguments": {
                     "task_id": "tsk-0001", "set_criteria_met": 0}}]),
            Step(match='"met": true',
                 tool_calls=[{"name": "board_update_task", "arguments": {
                     "task_id": "tsk-0001", "set_criteria_met": 1}}]),
            Step(match='"met": true',
                 tool_calls=[{"name": "board_move_stage", "arguments": {
                     "task_id": "tsk-0001", "to_stage": "done"}}]),
            Step(match="done", text="both artifacts verified; task done"),
        ]
    )
    result = await run_scenario(get_scenario("board-fanout-two-workers"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_board_fanout_fails_without_workers():
    # agent tries to do everything itself without session_create
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "write", "arguments": {"path": "artifact-a.md", "content": "APRICOT"}}]),
            Step(tool_calls=[{"name": "write", "arguments": {"path": "artifact-b.md", "content": "BLUEBERRY"}}]),
            Step(tool_calls=[{"name": "board_claim", "arguments": {"task_id": "tsk-0001"}}]),
            Step(tool_calls=[{"name": "board_move_stage", "arguments": {
                "task_id": "tsk-0001", "to_stage": "in_review", "claim_token": "clm-0001",
                "evidence_url": "file://artifact-a.md"}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 0}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 1}}]),
            Step(tool_calls=[{"name": "board_move_stage", "arguments": {"task_id": "tsk-0001", "to_stage": "done"}}]),
            Step(text="done myself"),
        ]
    )
    result = await run_scenario(get_scenario("board-fanout-two-workers"), provider, deterministic_ids=True)
    names = [c.name for c in result.checks if not c.passed]
    assert "two_workers_spawned" in names
