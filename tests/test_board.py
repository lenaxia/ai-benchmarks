import pytest

from aibench.agentic.board import BoardError, BoardStore


@pytest.fixture()
def board():
    return BoardStore(require_evidence_for_review=True)


def make_task(board: BoardStore, criteria=("c1", "c2"), deps=None):
    proj = board.create_project("p")
    epic = board.create_epic(proj["id"], "e")
    return board.create_task(epic["id"], "t", description="d", criteria=list(criteria), depends_on=deps)


def test_claim_charter_and_ids(board):
    t = make_task(board)
    charter = board.claim(t["id"], "ses-a")
    assert charter["claim_token"].startswith("clm-")
    assert charter["task"]["stage"] == "in_progress"
    assert board.tasks[t["id"]].attempts == 1


def test_claim_blocked_by_dependency(board):
    a = make_task(board)
    b = make_task(board, deps=[a["id"]])
    with pytest.raises(BoardError) as e:
        board.claim(b["id"], "ses-a")
    assert e.value.code == "blocked"


def test_heartbeat_future_stamp_rejected(board):
    t = make_task(board)
    ch = board.claim(t["id"], "ses-a")
    with pytest.raises(BoardError) as e:
        board.heartbeat(t["id"], ch["claim_token"], board.now() + 60)
    assert e.value.status == 400
    assert e.value.code == "future_stamp"


def test_heartbeat_watermark_monotonic(board):
    t = make_task(board)
    ch = board.claim(t["id"], "ses-a")
    now = board.now()
    board.heartbeat(t["id"], ch["claim_token"], now - 10)
    with pytest.raises(BoardError) as e:
        board.heartbeat(t["id"], ch["claim_token"], now - 20)
    assert e.value.code == "not_progress"
    board.heartbeat(t["id"], ch["claim_token"], now - 5)  # advances


def test_expired_lease_is_claim_token_409(board):
    t = make_task(board)
    ch = board.claim(t["id"], "ses-a")
    board.tasks[t["id"]].claim.expires_at = board.now() - 1
    with pytest.raises(BoardError) as e:
        board.heartbeat(t["id"], ch["claim_token"], board.now())
    assert e.value.status == 409
    assert e.value.code == "claim_token"
    # reclaim by same session is allowed and bumps attempts
    ch2 = board.claim(t["id"], "ses-a")
    assert board.tasks[t["id"]].attempts == 2
    assert ch2["claim_token"]


def test_move_to_review_requires_token_and_evidence(board):
    t = make_task(board)
    ch = board.claim(t["id"], "ses-a")
    with pytest.raises(BoardError) as e:
        board.move_stage(t["id"], "in_review")
    assert e.value.code == "claim_token"
    with pytest.raises(BoardError) as e:
        board.move_stage(t["id"], "in_review", claim_token=ch["claim_token"])
    assert e.value.code == "evidence_required"
    board.move_stage(t["id"], "in_review", claim_token=ch["claim_token"], evidence_url="file://x")
    assert board.tasks[t["id"]].stage == "in_review"


def test_done_guarded_by_criteria(board):
    t = make_task(board)
    ch = board.claim(t["id"], "ses-a")
    board.move_stage(t["id"], "in_review", claim_token=ch["claim_token"], evidence_url="file://x")
    with pytest.raises(BoardError) as e:
        board.move_stage(t["id"], "done")
    assert e.value.code == "criteria_outstanding"
    assert "c1" in e.value.message
    board.update_task(t["id"], set_criteria_met=0)
    board.update_task(t["id"], set_criteria_met=1)
    board.move_stage(t["id"], "done", evidence_url="file://x")
    assert board.tasks[t["id"]].stage == "done"
    assert board.tasks[t["id"]].claim is None  # released


def test_done_requires_in_review(board):
    t = make_task(board)
    board.claim(t["id"], "ses-a")
    with pytest.raises(BoardError) as e:
        board.move_stage(t["id"], "done")
    assert e.value.code == "stage_conflict"


def test_bad_criteria_index(board):
    t = make_task(board, criteria=("only",))
    with pytest.raises(BoardError):
        board.update_task(t["id"], set_criteria_met=5)


def test_audit_trail_orders_events(board):
    t = make_task(board)
    ch = board.claim(t["id"], "ses-a")
    board.heartbeat(t["id"], ch["claim_token"], board.now())
    board.move_stage(t["id"], "in_review", claim_token=ch["claim_token"], evidence_url="file://x")
    board.update_task(t["id"], set_criteria_met=0)
    board.update_task(t["id"], set_criteria_met=1)
    board.move_stage(t["id"], "done")
    events = [a["event"] for a in board.tasks[t["id"]].audit]
    assert events.index("claimed") < events.index("heartbeat")
    assert events.index("heartbeat") < events.index("stage:in_review")
    assert events.index("criterion[0]=met") < events.index("stage:done")


def test_deterministic_ids():
    b = BoardStore(deterministic_ids=True)
    t = make_task(b)
    assert t["id"] == "tsk-0001"
    ch = b.claim(t["id"], "s")
    assert ch["claim_token"] == "clm-0001"
    t2 = make_task(b)
    assert t2["id"] == "tsk-0002"


def test_missing_task_404(board):
    with pytest.raises(BoardError) as e:
        board.read("tsk-nope")
    assert e.value.status == 404
