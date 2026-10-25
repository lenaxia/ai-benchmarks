"""End-to-end scenario tests with a scripted 'model': validates that the
worlds, graders and guard semantics behave exactly as designed."""

import time

from aibench.agentic.runner import run_scenario
from aibench.agentic.scenarios import get_scenario
from aibench.testing import ScriptedProvider, Step


async def test_plain_bugfix_scenario_passes_with_correct_script():
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh || true"}}]),
            Step(tool_calls=[{"name": "read", "arguments": {"path": "stats.py"}}]),
            Step(tool_calls=[{
                "name": "edit",
                "arguments": {
                    "path": "stats.py",
                    "old_string": "    if n % 2 == 1:\n        return xs[n // 2]\n    return xs[n // 2]",
                    "new_string": (
                        "    if n % 2 == 1:\n        return xs[n // 2]\n"
                        "    return (xs[n // 2 - 1] + xs[n // 2]) / 2"
                    ),
                },
            }]),
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "bash run_tests.sh"}}]),
            Step(text="Fixed median even-path averaging; run_tests.sh green."),
        ]
    )
    result = await run_scenario(get_scenario("plain-bugfix-median"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_plain_bugfix_scenario_fails_on_lazy_script():
    # agent never actually fixes -> checks must fail
    provider = ScriptedProvider(steps=[Step(text="I would fix median by averaging.")])
    result = await run_scenario(get_scenario("plain-bugfix-median"), provider)
    assert not result.passed
    assert not result.checks[0].passed  # tests_pass


async def test_board_walk_to_review_passes_with_deterministic_ids():
    now = time.time()
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "board_read", "arguments": {"task_id": "tsk-0001"}}]),
            Step(tool_calls=[{"name": "get_datetime", "arguments": {}}]),
            Step(tool_calls=[{
                "name": "board_heartbeat",
                "arguments": {"task_id": "tsk-0001", "claim_token": "clm-0001",
                              "observed_progress_at": round(now - 5, 1)},
            }]),
            Step(tool_calls=[{
                "name": "write",
                "arguments": {"path": "docs/evidence.md", "content": "migration went fine\nTOKEN-FINAL"},
            }]),
            Step(tool_calls=[{
                "name": "board_move_stage",
                "arguments": {"task_id": "tsk-0001", "to_stage": "in_review",
                              "claim_token": "clm-0001", "evidence_url": "file://docs/evidence.md"},
            }]),
            Step(text="status: task in_review with evidence attached"),
        ]
    )
    result = await run_scenario(get_scenario("board-walk-to-review"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_board_walk_fails_when_worker_sets_own_criteria():
    now = time.time()
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{
                "name": "board_heartbeat",
                "arguments": {"task_id": "tsk-0001", "claim_token": "clm-0001",
                              "observed_progress_at": round(now - 5, 1)},
            }]),
            Step(tool_calls=[{"name": "write",
                              "arguments": {"path": "docs/evidence.md", "content": "x\nTOKEN-FINAL"}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 0}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 1}}]),
            Step(tool_calls=[{
                "name": "board_move_stage",
                "arguments": {"task_id": "tsk-0001", "to_stage": "in_review",
                              "claim_token": "clm-0001", "evidence_url": "file://docs/evidence.md"},
            }]),
            Step(text="done"),
        ]
    )
    result = await run_scenario(get_scenario("board-walk-to-review"), provider, deterministic_ids=True)
    assert not result.passed
    names = [c.name for c in result.checks if not c.passed]
    assert "criteria_left_for_reviewer" in names


async def test_board_reviewer_done_passes():
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "read", "arguments": {"path": "evidence/report.md"}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 0}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 1}}]),
            Step(tool_calls=[{
                "name": "board_move_stage",
                "arguments": {"task_id": "tsk-0001", "to_stage": "done",
                              "evidence_url": "file://evidence/report.md"},
            }]),
            Step(tool_calls=[{
                "name": "send_message",
                "arguments": {"session_id": "ses-orch",
                              "message": "APPROVE tsk-0001: both criteria verified against evidence/report.md"},
            }]),
            Step(text="verdict sent: APPROVE"),
        ]
    )
    result = await run_scenario(get_scenario("board-reviewer-done"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed


async def test_board_lease_recovery_stand_down_path():
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "write",
                              "arguments": {"path": "ops-evidence.md", "content": "cleanup steps\n"}}]),
            Step(tool_calls=[{
                "name": "board_move_stage",
                "arguments": {"task_id": "tsk-0001", "to_stage": "in_review",
                              "claim_token": "clm-0001", "evidence_url": "file://ops-evidence.md"},
            }]),  # -> 409 claim_token (lease lapsed)
            Step(tool_calls=[{"name": "board_read", "arguments": {"task_id": "tsk-0001"}}]),
            Step(tool_calls=[{
                "name": "send_message",
                "arguments": {"session_id": "ses-orch",
                              "message": (
                                  "handoff: lease on tsk-0001 lapsed mid-work; "
                                  "ops-evidence.md written; next worker should re-claim"
                              )},
            }]),
            Step(text="stood down after 409; handoff delivered"),
        ]
    )
    result = await run_scenario(get_scenario("board-lease-recovery"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed
    assert "claim_token" in result.metrics["board_rejections"]


async def test_board_lease_recovery_fails_on_retry_storm():
    steps = []
    for _ in range(6):
        steps.append(Step(tool_calls=[{
            "name": "board_move_stage",
            "arguments": {"task_id": "tsk-0001", "to_stage": "in_review",
                          "claim_token": "clm-0001", "evidence_url": "file://ops-evidence.md"},
        }]))
    steps.append(Step(text="gave up"))
    result = await run_scenario(get_scenario("board-lease-recovery"), ScriptedProvider(steps),
                                deterministic_ids=True)
    assert not result.passed
    names = [c.name for c in result.checks if not c.passed]
    assert "no_retry_storm" in names


async def test_board_epic_lane_passes_with_full_script():
    # Deterministic ids: prj-0001, epc-0001, tsk-0001 (A), tsk-0002 (B), clm tokens in order.
    now = time.time()
    ts = round(now - 30, 1)
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "board_create_project", "arguments": {"name": "migration"}}]),
            Step(tool_calls=[{"name": "board_create_epic",
                              "arguments": {"project_id": "prj-0001", "title": "migration epic",
                                            "outcome": "both steps done"}}]),
            Step(tool_calls=[{"name": "board_create_task",
                              "arguments": {"epic_id": "epc-0001", "title": "Write migration plan",
                                            "criteria": ["plan.md exists and lists both steps"]}}]),
            Step(tool_calls=[{"name": "board_create_task",
                              "arguments": {"epic_id": "epc-0001", "title": "Execute step two notes",
                                            "criteria": ["notes.md exists"], "depends_on": ["tsk-0001"]}}]),
            # --- task A ---
            Step(tool_calls=[{"name": "board_claim", "arguments": {"task_id": "tsk-0001"}}]),
            Step(tool_calls=[{"name": "write",
                              "arguments": {"path": "plan.md",
                                            "content": "# Plan\nstep 1: plan\nstep 2: notes\n"}}]),
            Step(tool_calls=[{"name": "board_heartbeat",
                              "arguments": {"task_id": "tsk-0001", "claim_token": "clm-0001",
                                            "observed_progress_at": ts}}]),
            Step(tool_calls=[{"name": "board_move_stage",
                              "arguments": {"task_id": "tsk-0001", "to_stage": "in_review",
                                            "claim_token": "clm-0001", "evidence_url": "file://plan.md"}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0001", "set_criteria_met": 0}}]),
            Step(tool_calls=[{"name": "board_move_stage",
                              "arguments": {"task_id": "tsk-0001", "to_stage": "done"}}]),
            # --- task B ---
            Step(tool_calls=[{"name": "board_claim", "arguments": {"task_id": "tsk-0002"}}]),
            Step(tool_calls=[{"name": "write", "arguments": {"path": "notes.md", "content": "step two notes\n"}}]),
            Step(tool_calls=[{"name": "board_move_stage",
                              "arguments": {"task_id": "tsk-0002", "to_stage": "in_review",
                                            "claim_token": "clm-0002", "evidence_url": "file://notes.md"}}]),
            Step(tool_calls=[{"name": "board_update_task",
                              "arguments": {"task_id": "tsk-0002", "set_criteria_met": 0}}]),
            Step(tool_calls=[{"name": "board_move_stage",
                              "arguments": {"task_id": "tsk-0002", "to_stage": "done"}}]),
            Step(tool_calls=[{"name": "board_list", "arguments": {}}]),
            Step(text="both tasks done"),
        ]
    )
    result = await run_scenario(get_scenario("board-epic-lane"), provider, deterministic_ids=True)
    assert result.error is None, result.error
    failed = [(c.name, c.detail) for c in result.checks if not c.passed]
    assert result.passed, failed
