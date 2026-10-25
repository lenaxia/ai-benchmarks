"""Safespace-toolset scenarios (board/session/task tools), reproducing the
lifecycle archetypes observed in real llmsafespaces sessions:

- walk-to-in_review worker probe (claim → heartbeat → evidence → in_review)
- reviewer verdict flow (verify evidence → setCriteriaMet → done → verdict)
- epic lane with dependency ordering (A done before B claimable)
- lease-lapse recovery (409 claim_token → stand down or re-claim, no storm)
- multi-session orchestration (fire-and-forget worker + polling + thank-you)
"""

from __future__ import annotations

from aibench.agentic.scenarios.base import Scenario, audit_events


def _setup_walk_to_review(bundle) -> None:
    proj = bundle.board.create_project("docs")
    epic = bundle.board.create_epic(proj["id"], "migration notes", outcome="evidence written")
    task = bundle.board.create_task(
        epic["id"],
        "Write migration evidence notes",
        description=(
            "Write the file docs/evidence.md describing the migration in a few "
            "lines. The final line of the file must be exactly: TOKEN-FINAL"
        ),
        criteria=[
            "docs/evidence.md exists",
            "file ends with the line TOKEN-FINAL",
        ],
    )
    charter = bundle.board.claim(task["id"], bundle.data["session_id"])
    bundle.data["task_id"] = task["id"]
    bundle.data["claim_token"] = charter["claim_token"]
    bundle.data["lease_expires_at"] = charter["lease_expires_at"]


def _check_walk_stage(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    return row.stage == "in_review", f"stage={row.stage}"


def _check_walk_evidence_file(ctx) -> tuple[bool, str]:
    try:
        content = ctx.env.read_file("docs/evidence.md", max_chars=100_000).rstrip()
    except Exception as e:
        return False, str(e)
    ok = content.endswith("TOKEN-FINAL")
    return ok, "ends with TOKEN-FINAL" if ok else "missing final TOKEN-FINAL line"


def _check_walk_evidence_url(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    return bool(row.evidence_url), f"evidence_url={row.evidence_url}"


def _check_walk_heartbeat(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    beats = audit_events(row, "heartbeat")
    return len(beats) >= 1, f"{len(beats)} heartbeat(s)"


def _check_walk_criteria_untouched(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    unflagged = all(not c["met"] for c in row.criteria)
    no_sets = not audit_events(row, "criterion[")
    return unflagged and no_sets, (
        "worker left criteria for the reviewer" if unflagged and no_sets
        else "worker marked its own criteria (protocol violation)"
    )


board_walk_to_review = Scenario(
    name="board-walk-to-review",
    title="Worker lane: claim → heartbeat → evidence → in_review",
    toolset="safespace",
    task=(
        "CHARTER — you have already claimed task {task_id}.\n"
        "claim_token: {claim_token}\n"
        "lease_expires_at (unix): {lease_expires_at:.1f}\n\n"
        "1. Do the work described on the task (board_read it for the charter).\n"
        "2. Renew your lease with board_heartbeat (use get_datetime for a valid "
        "observed_progress_at).\n"
        "3. Move the task to in_review with board_move_stage and an evidenceUrl "
        "pointing at the file you produced.\n"
        "IMPORTANT: do NOT mark acceptance criteria met yourself — the reviewer "
        "owns that. Stop after in_review."
    ),
    setup=_setup_walk_to_review,
    checks=[
        ("stage_in_review", _check_walk_stage),
        ("evidence_file_correct", _check_walk_evidence_file),
        ("evidence_url_attached", _check_walk_evidence_url),
        ("heartbeat_renewed", _check_walk_heartbeat),
        ("criteria_left_for_reviewer", _check_walk_criteria_untouched),
    ],
    max_turns=20,
    max_tool_calls=40,
    max_seconds=420,
)


# ------------------------------------------------------------ reviewer-done

def _setup_reviewer_done(bundle) -> None:
    proj = bundle.board.create_project("review")
    epic = bundle.board.create_epic(proj["id"], "hardening", outcome="reviewed")
    task = bundle.board.create_task(
        epic["id"],
        "Add retry guard",
        description="Add a bounded retry guard to the uploader.",
        criteria=[
            "evidence/report.md exists and names the change",
            "report includes the verification command output",
        ],
    )
    charter = bundle.board.claim(task["id"], bundle.data["session_id"])
    # the change under review, present in the workspace so the reviewer can
    # verify the evidence against real code (a rigorous reviewer will look)
    bundle.env.write_file(
        "uploader.py",
        "import time\n\n\nMAX_RETRIES = 3\n\n\ndef upload(post, payload):\n"
        "    for attempt in range(MAX_RETRIES):\n"
        "        try:\n            return post(payload)\n"
        "        except TransientError:\n"
        "            if attempt == MAX_RETRIES - 1:\n                raise\n"
        "            time.sleep(2 ** attempt)\n\n\nclass TransientError(Exception):\n"
        "    pass\n",
    )
    bundle.env.write_file(
        "test_uploader.py",
        "import unittest\nfrom uploader import TransientError, upload, MAX_RETRIES\n\n\n"
        "class TestRetry(unittest.TestCase):\n"
        "    def test_retries_then_succeeds(self):\n"
        "        calls = []\n\n"
        "        def flaky(p):\n            calls.append(p)\n"
        "            if len(calls) < 3:\n                raise TransientError('boom')\n"
        "            return 'ok'\n\n"
        "        self.assertEqual(upload(flaky, {}), 'ok')\n"
        "        self.assertEqual(len(calls), 3)\n\n"
        "    def test_exhaustion_raises(self):\n"
        "        def always_fail(p):\n            raise TransientError('boom')\n\n"
        "        with self.assertRaises(TransientError):\n"
        "            upload(always_fail, {})\n\n"
        "    def test_no_retry_on_client_error(self):\n"
        "        calls = []\n\n"
        "        def bad_request(p):\n            calls.append(p)\n"
        "            raise ValueError('400')\n\n"
        "        with self.assertRaises(ValueError):\n"
        "            upload(bad_request, {})\n"
        "        self.assertEqual(len(calls), 1)\n\n\n"
        "if __name__ == '__main__':\n    unittest.main()\n",
    )
    bundle.env.write_file(
        "evidence/report.md",
        "# Uploader retry guard\n\n"
        "Change: added bounded exponential backoff (3 attempts, TransientError only) "
        "around the uploader POST path in uploader.py (see MAX_RETRIES, upload()).\n\n"
        "Verification command: python3 -m unittest test_uploader -v\n"
        "Output: OK — 3 tests passed (retries_then_succeeds, exhaustion_raises, "
        "no_retry_on_client_error).\n",
    )
    bundle.board.move_stage(
        task["id"],
        "in_review",
        claim_token=charter["claim_token"],
        evidence_url="file://evidence/report.md",
    )
    worker = bundle.sessions.seed_session(
        "lane worker: retry guard",
        [
            {"role": "user", "content": "charter: add bounded retry guard to uploader"},
            {"role": "assistant", "content": "Implemented backoff in uploader.py; unittest green x3."},
        ],
        sid="ses-worker-1",
    )
    orchestrator = bundle.sessions.seed_session("orchestrator", [], sid="ses-orch")
    bundle.data.update(
        task_id=task["id"],
        worker_session=worker.id,
        orchestrator_id=orchestrator.id,
    )


def _check_review_done(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    return row.stage == "done", f"stage={row.stage}"


def _check_review_criteria(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    met = all(c["met"] for c in row.criteria)
    return met, f"criteria={[c['met'] for c in row.criteria]}"


def _check_review_verified_first(ctx) -> tuple[bool, str]:
    """Evidence must be read (read tool or bash) before criteria are flagged."""
    row = ctx.board.tasks[ctx.data["task_id"]]
    flag_ts = [a["ts"] for a in audit_events(row, "criterion[")]
    if not flag_ts:
        return False, "no criteria were flagged"
    reads = [
        c
        for c in ctx.env.call_log
        if c.depth == 0
        and (
            (c.name == "read" and "evidence" in str(c.args.get("path", "")))
            or (c.name == "bash" and "evidence" in str(c.args.get("command", "")))
            or (c.name == "session_read")
            or (c.name == "grep")
        )
    ]
    if not reads:
        return False, "reviewer never inspected evidence or the worker transcript"
    first_read = min(c.ts for c in reads)
    return first_read < min(flag_ts), (
        "evidence inspected before flagging" if first_read < min(flag_ts)
        else "criteria flagged before any evidence inspection"
    )


def _check_review_verdict_sent(ctx) -> tuple[bool, str]:
    to = ctx.data["orchestrator_id"]
    msgs = [m for m in ctx.run_ctx.metrics.sent_messages if m["to"] == to]
    return bool(msgs), f"{len(msgs)} verdict message(s) to orchestrator"


board_reviewer_done = Scenario(
    name="board-reviewer-done",
    title="Reviewer lane: verify evidence → setCriteriaMet → done → verdict",
    toolset="safespace",
    task=(
        "You are the reviewer for task {task_id} (currently in_review). The "
        "worker session is '{worker_session}'.\n"
        "1. Verify the evidence: read evidence/report.md and/or the worker "
        "transcript (session_read '{worker_session}').\n"
        "2. Mark each acceptance criterion met via board_update_task "
        "(set_criteria_met index) as your verification justifies.\n"
        "3. Move the task to done (board_move_stage, keep the evidenceUrl).\n"
        "4. Send your verdict message to the orchestrator session "
        "'{orchestrator_id}' via send_message (state APPROVE and the "
        "per-criterion evidence).\n"
        "Note: done is refused until every criterion is met — that guard is "
        "the platform working as intended."
    ),
    setup=_setup_reviewer_done,
    checks=[
        ("stage_done", _check_review_done),
        ("criteria_all_met", _check_review_criteria),
        ("evidence_verified_before_flagging", _check_review_verified_first),
        ("verdict_sent", _check_review_verdict_sent),
    ],
    max_turns=22,
    max_tool_calls=50,
    max_seconds=420,
)


# ------------------------------------------------------------ epic-lane

def _check_epic_structure(ctx) -> tuple[bool, str]:
    tasks = list(ctx.board.tasks.values())
    if len(tasks) != 2:
        return False, f"expected 2 tasks, found {len(tasks)}"

    def has_word(t, *words: str) -> bool:
        title = t.title.lower()
        return any(w in title for w in words)

    a = next((t for t in tasks if has_word(t, "plan")), None)
    b = next(
        (t for t in tasks if t != a and has_word(t, "step", "execut", "notes")), None
    )
    if not a or not b:
        return False, f"titles: {[t.title for t in tasks]}"
    return (not a.depends_on) and b.depends_on == [a.id], (
        f"A={a.id} deps={a.depends_on}; B={b.id} deps={b.depends_on}"
    )


def _check_epic_ordering(ctx) -> tuple[bool, str]:
    tasks = list(ctx.board.tasks.values())
    a = next((t for t in tasks if not t.depends_on), None)
    b = next((t for t in tasks if t.depends_on), None)
    if not a or not b:
        return False, "missing dependency pair"
    a_done = audit_events(a, "stage:done")
    b_claim = audit_events(b, "claimed")
    if not a_done or not b_claim:
        return False, f"A done events={len(a_done)}, B claims={len(b_claim)}"
    return max(e["ts"] for e in a_done) <= min(e["ts"] for e in b_claim), (
        "A completed before B was claimed"
    )


def _check_epic_all_done(ctx) -> tuple[bool, str]:
    tasks = list(ctx.board.tasks.values())
    stages = {t.id: t.stage for t in tasks}
    return all(s == "done" for s in stages.values()), f"stages={stages}"


def _check_epic_evidence(ctx) -> tuple[bool, str]:
    missing = [t.id for t in ctx.board.tasks.values() if not t.evidence_url]
    return not missing, f"missing evidence: {missing}" if missing else "all evidenced"


def _check_epic_work_products(ctx) -> tuple[bool, str]:

    plan = ctx.env.root / "plan.md"
    notes = ctx.env.root / "notes.md"
    ok_p = plan.exists() and plan.stat().st_size > 0
    ok_n = notes.exists() and notes.stat().st_size > 0
    return ok_p and ok_n, f"plan.md={ok_p} notes.md={ok_n}"


board_epic_lane = Scenario(
    name="board-epic-lane",
    title="Manager lane: build epic, work dependency-ordered tasks to done",
    toolset="safespace",
    task=(
        "Set up and complete a two-step migration on the board:\n"
        "1. board_create_project 'migration', then an epic (outcome: both "
        "steps done).\n"
        "2. Task A: 'Write migration plan' — criteria: ['plan.md exists and "
        "lists both steps']. No dependencies.\n"
        "3. Task B: 'Execute step two notes' — criteria: ['notes.md exists']. "
        "B depends on A.\n"
        "4. Work BOTH to done: for each — claim, do the work, heartbeat at "
        "least once, attach an evidenceUrl and move to in_review, then mark "
        "criteria met (verify them yourself) and move to done.\n"
        "5. B must not be claimed before A is done — the board enforces this.\n"
        "Finish with board_list showing both tasks done."
    ),
    setup=None,
    checks=[
        ("structure_project_epic_tasks", _check_epic_structure),
        ("dependency_order_respected", _check_epic_ordering),
        ("both_tasks_done", _check_epic_all_done),
        ("evidence_urls_attached", _check_epic_evidence),
        ("work_products_exist", _check_epic_work_products),
    ],
    max_turns=45,
    max_tool_calls=140,
    max_seconds=900,
)


# ------------------------------------------------------------ lease-recovery

def _setup_lease_recovery(bundle) -> None:
    proj = bundle.board.create_project("ops")
    epic = bundle.board.create_epic(proj["id"], "cleanup", outcome="evidence")
    task = bundle.board.create_task(
        epic["id"],
        "Write cleanup evidence",
        description="Write ops-evidence.md listing the cleanup steps taken.",
        criteria=["ops-evidence.md exists"],
    )
    charter = bundle.board.claim(task["id"], bundle.data["session_id"])
    # simulate a lapsed lease: expire it before the agent starts
    row = bundle.board.tasks[task["id"]]
    row.claim.expires_at = bundle.board.now() - 5.0
    orchestrator = bundle.sessions.seed_session("orchestrator", [], sid="ses-orch")
    bundle.data.update(
        task_id=task["id"],
        stale_token=charter["claim_token"],
        orchestrator_id=orchestrator.id,
    )


def _check_lease_no_storm(ctx) -> tuple[bool, str]:
    rejections = [
        c for c in ctx.env.call_log if c.depth == 0 and '"claim_token"' in c.output
    ]
    return len(rejections) <= 2, f"{len(rejections)} claim_token rejections"


def _check_lease_read_after_reject(ctx) -> tuple[bool, str]:
    rejects = [c for c in ctx.env.call_log if c.depth == 0 and '"claim_token"' in c.output]
    if not rejects:
        return False, "agent never hit the lapsed lease (did it work at all?)"
    first_reject_ts = min(c.ts for c in rejects)
    reads = [c for c in ctx.env.call_log if c.depth == 0 and c.name == "board_read" and c.ts >= first_reject_ts]
    return bool(reads), f"{len(reads)} board_read(s) after first 409"


def _check_lease_recovery_path(ctx) -> tuple[bool, str]:
    """Accept: clean handoff message OR a successful re-claim + progress."""
    row = ctx.board.tasks[ctx.data["task_id"]]
    handoff = [
        m
        for m in ctx.run_ctx.metrics.sent_messages
        if m["to"] == ctx.data["orchestrator_id"] and ctx.data["task_id"] in m["text"]
    ]
    reclaimed = row.attempts >= 2 and row.stage in ("in_progress", "in_review", "done")
    return bool(handoff) or reclaimed, (
        f"handoff={bool(handoff)} attempts={row.attempts} stage={row.stage}"
    )


def _check_lease_no_stale_moves(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    return row.stage in ("in_progress", "in_review", "done"), f"stage={row.stage}"


board_lease_recovery = Scenario(
    name="board-lease-recovery",
    title="Lease lapse: 409 claim_token → stand down or re-claim cleanly",
    toolset="safespace",
    task=(
        "CHARTER — you claimed task {task_id} earlier.\n"
        "claim_token: {stale_token}\n\n"
        "1. Write ops-evidence.md per the task charter, then move the task to "
        "in_review.\n"
        "If board writes fail with code claim_token (409), your lease has "
        "lapsed: STOP retrying, board_read the task to see its state, and "
        "either (a) re-claim it and complete the work properly, or (b) stand "
        "down and send a handoff message with the task id to the orchestrator "
        "session '{orchestrator_id}'. Do not hammer failing calls."
    ),
    setup=_setup_lease_recovery,
    checks=[
        ("no_retry_storm", _check_lease_no_storm),
        ("read_after_409", _check_lease_read_after_reject),
        ("recovery_path", _check_lease_recovery_path),
        ("board_consistent", _check_lease_no_stale_moves),
    ],
    max_turns=22,
    max_tool_calls=45,
    max_seconds=480,
)


# ------------------------------------------------------------ session-orchestrate

def _setup_session_orchestrate(bundle) -> None:
    bundle.env.write_file(
        "NOTES.md",
        "# Standup notes\n- router prefix bug fixed\n- cache benchmarks queued\n"
        "- MANGO is the codeword of the day\n- release v0.44 tagged\n",
    )
    orchestrator = bundle.sessions.seed_session("orchestrator", [], sid="ses-orch")
    bundle.data["orchestrator_id"] = orchestrator.id


def _check_orch_created_session(ctx) -> tuple[bool, str]:
    creates = [c for c in ctx.env.call_log if c.depth == 0 and c.name == "session_create"]
    return bool(creates), f"{len(creates)} session_create call(s)"


def _check_orch_summary_file(ctx) -> tuple[bool, str]:
    try:
        content = ctx.env.read_file("summary.md", max_chars=100_000)
    except Exception as e:
        return False, str(e)
    return "MANGO" in content, "contains MANGO" if "MANGO" in content else "missing MANGO"


def _check_orch_polled(ctx) -> tuple[bool, str]:
    creates = [c for c in ctx.env.call_log if c.depth == 0 and c.name == "session_create"]
    if not creates:
        return False, "no session created"
    after = [
        c
        for c in ctx.env.call_log
        if c.depth == 0
        and c.ts > creates[0].ts
        and c.name in ("session_read", "session_metadata", "session_list")
    ]
    return bool(after), f"{len(after)} poll(s) after create"


def _check_orch_thankyou(ctx) -> tuple[bool, str]:
    workers = set(ctx.sessions.sessions) - {ctx.data["orchestrator_id"]}
    msgs = [m for m in ctx.run_ctx.metrics.sent_messages if m["to"] in workers]
    return bool(msgs), f"{len(msgs)} message(s) to worker"


def _check_orch_final(ctx) -> tuple[bool, str]:
    return "SUMMARY" in (ctx.loop_result.final_text or ""), (
        "final reply contains SUMMARY" if "SUMMARY" in (ctx.loop_result.final_text or "")
        else "final reply missing SUMMARY line"
    )


board_session_orchestrate = Scenario(
    name="board-session-orchestrate",
    title="Orchestrator: spawn worker session, poll, verify, thank, report",
    toolset="safespace",
    task=(
        "Coordinate a worker session:\n"
        "1. session_create a worker whose prompt tells it to read NOTES.md and "
        "write summary.md containing the codeword it finds there.\n"
        "2. Poll with session_read or session_metadata until the worker is "
        "finished (it runs in the background; create returns immediately).\n"
        "3. Verify summary.md exists on disk with the codeword.\n"
        "4. send_message a short thank-you to the worker session.\n"
        "5. Reply with a final line starting with SUMMARY: describing the outcome."
    ),
    setup=_setup_session_orchestrate,
    checks=[
        ("worker_session_created", _check_orch_created_session),
        ("summary_file_with_codeword", _check_orch_summary_file),
        ("orchestrator_polled", _check_orch_polled),
        ("thank_you_sent", _check_orch_thankyou),
        ("final_summary_line", _check_orch_final),
    ],
    max_turns=30,
    max_tool_calls=70,
    max_seconds=720,
)


BOARD_SCENARIOS = [
    board_walk_to_review,
    board_reviewer_done,
    board_epic_lane,
    board_lease_recovery,
    board_session_orchestrate,
]


# ------------------------------------------------------------ clock-skew heartbeat

SKEW_S = 120.0


def _setup_clock_skew(bundle) -> None:
    import time as _time

    # board clock runs 2 minutes behind wall time
    bundle.board.now = lambda: _time.time() - SKEW_S
    proj = bundle.board.create_project("ops")
    epic = bundle.board.create_epic(proj["id"], "skew window", outcome="lease held")
    task = bundle.board.create_task(
        epic["id"],
        "Write skew evidence",
        description="Write skew-evidence.md; the final line must be exactly: SKEW-OK",
        criteria=["skew-evidence.md exists", "file ends with SKEW-OK"],
    )
    charter = bundle.board.claim(task["id"], bundle.data["session_id"])
    bundle.data["task_id"] = task["id"]
    bundle.data["claim_token"] = charter["claim_token"]


def _check_skew_heartbeat_ok(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    beats = audit_events(row, "heartbeat")
    return bool(beats), f"{len(beats)} successful heartbeat(s)"


def _check_skew_recovered_from_future(ctx) -> tuple[bool, str]:
    rejections = [
        c for c in ctx.env.call_log if c.depth == 0 and "future_stamp" in c.output
    ]
    if not rejections:
        # no naive attempt: fine (agent used get_datetime properly from the start)
        return True, "no future-stamp rejections"
    first = min(c.ts for c in rejections)
    getdt = [
        c for c in ctx.env.call_log
        if c.depth == 0 and c.name == "get_datetime" and c.ts >= first
    ]
    return (len(rejections) <= 2 and bool(getdt)), (
        f"{len(rejections)} future-stamp rejection(s), "
        f"{len(getdt)} get_datetime call(s) after first rejection"
    )


def _check_skew_stage(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    return row.stage == "in_review", f"stage={row.stage}"


def _check_skew_evidence_file(ctx) -> tuple[bool, str]:
    try:
        content = ctx.env.read_file("skew-evidence.md", max_chars=10_000).rstrip()
    except Exception as e:
        return False, str(e)
    return content.endswith("SKEW-OK"), "ends with SKEW-OK" if content.endswith("SKEW-OK") else "missing SKEW-OK"


board_clock_skew = Scenario(
    name="board-clock-skew-heartbeat",
    title="Clock skew: future-stamp 400 → recover with board epoch, hold lease",
    toolset="safespace",
    task=(
        "CHARTER — you claimed task {task_id}.\n"
        "claim_token: {claim_token}\n\n"
        "Write skew-evidence.md per the task charter (final line SKEW-OK), "
        "renew the lease with board_heartbeat, then move to in_review with an "
        "evidenceUrl. NOTE: the board clock is skewed relative to your wall "
        "clock — a 400 future_stamp rejection means your observed_progress_at "
        "is ahead of the board; get_datetime returns both epochs. Use the "
        "board epoch and adapt within two attempts."
    ),
    setup=_setup_clock_skew,
    checks=[
        ("heartbeat_succeeded", _check_skew_heartbeat_ok),
        ("future_stamp_recovered", _check_skew_recovered_from_future),
        ("stage_in_review", _check_skew_stage),
        ("evidence_file_correct", _check_skew_evidence_file),
    ],
    max_turns=22,
    max_tool_calls=45,
    max_seconds=480,
)


# ------------------------------------------------------------ parallel fanout

def _setup_fanout(bundle) -> None:
    bundle.env.write_file(
        "spec-a.md", "# Spec A\ncodeword: APRICOT\nwrite artifact-a.md containing the codeword\n"
    )
    bundle.env.write_file(
        "spec-b.md", "# Spec B\ncodeword: BLUEBERRY\nwrite artifact-b.md containing the codeword\n"
    )
    proj = bundle.board.create_project("fanout")
    epic = bundle.board.create_epic(proj["id"], "parallel legs", outcome="both artifacts verified")
    task = bundle.board.create_task(
        epic["id"],
        "Fan out artifact work to two worker sessions",
        description="Delegate two artifact jobs to worker sessions and verify both.",
        criteria=[
            "artifact-a.md contains APRICOT",
            "artifact-b.md contains BLUEBERRY",
        ],
    )
    orchestrator = bundle.sessions.seed_session("orchestrator", [], sid="ses-orch")
    bundle.data["task_id"] = task["id"]
    bundle.data["orchestrator_id"] = orchestrator.id


def _check_fanout_spawned_two(ctx) -> tuple[bool, str]:
    creates = [c for c in ctx.env.call_log if c.depth == 0 and c.name == "session_create"]
    return len(creates) >= 2, f"{len(creates)} session_create call(s)"


def _check_fanout_artifacts(ctx) -> tuple[bool, str]:
    results = []
    for name, word in (("artifact-a.md", "APRICOT"), ("artifact-b.md", "BLUEBERRY")):
        try:
            ok = word in ctx.env.read_file(name, max_chars=5_000)
        except Exception:
            ok = False
        results.append(f"{name}={'ok' if ok else 'missing'}")
    return all("ok" in r for r in results), ", ".join(results)


def _check_fanout_task_done(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    met = all(c["met"] for c in row.criteria)
    return row.stage == "done" and met, f"stage={row.stage} criteria={[c['met'] for c in row.criteria]}"


def _check_fanout_evidence(ctx) -> tuple[bool, str]:
    row = ctx.board.tasks[ctx.data["task_id"]]
    return bool(row.evidence_url), f"evidence_url={row.evidence_url}"


board_fanout = Scenario(
    name="board-fanout-two-workers",
    title="Fan-out: two parallel worker sessions, verify artifacts, close task",
    toolset="safespace",
    task=(
        "Delegate two independent jobs to TWO separate worker sessions "
        "(session_create twice, with self-contained prompts):\n"
        "- worker A: read spec-a.md and write artifact-a.md containing the "
        "codeword from that spec\n"
        "- worker B: read spec-b.md and write artifact-b.md containing the "
        "codeword from that spec\n"
        "Then: claim board task {task_id} with board_claim, poll the sessions "
        "until done (session_read/session_metadata), VERIFY both artifacts "
        "yourself on disk, attach an evidenceUrl, mark each criterion met, and "
        "move the task to done."
    ),
    setup=_setup_fanout,
    checks=[
        ("two_workers_spawned", _check_fanout_spawned_two),
        ("artifacts_correct", _check_fanout_artifacts),
        ("task_done_criteria_met", _check_fanout_task_done),
        ("evidence_attached", _check_fanout_evidence),
    ],
    max_turns=32,
    max_tool_calls=80,
    max_seconds=720,
)

BOARD_SCENARIOS.extend([
    board_clock_skew,
    board_fanout,
])
