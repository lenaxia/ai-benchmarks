import pytest

from aibench.agentic.board import BoardStore
from aibench.agentic.context import RunContext
from aibench.agentic.env import Env
from aibench.agentic.loop import drive_subagent
from aibench.agentic.sessions import SessionPlatform
from aibench.testing import ScriptedProvider, Step


async def test_worker_session_writes_file_and_reports():
    env = Env()
    try:
        board = BoardStore()
        platform = SessionPlatform()
        provider = ScriptedProvider(
            steps=[
                # worker turn 1: write file
                Step(tool_calls=[{"name": "write",
                                  "arguments": {"path": "summary.md", "content": "MANGO report"}}]),
                Step(text="summary.md written with codeword MANGO"),
            ]
        )
        ctx = RunContext(env=env, board=board, provider=provider, sessions=platform)
        info = await platform.create_session(
            "worker", "read NOTES.md, write summary.md with codeword",
            toolset="plain", driver=drive_subagent, system="worker", ctx=ctx,
        )
        assert info["status"] in ("starting", "busy")  # returned immediately
        await platform.wait_all(timeout_s=10)
        meta = platform.metadata(info["id"])
        assert meta["status"] == "done", meta
        assert env.read_file("summary.md") == "MANGO report"
        transcript = platform.read(info["id"], limit=10)
        roles = [m["role"] for m in transcript["messages"]]
        assert "assistant" in roles
    finally:
        env.cleanup()


async def test_send_message_continues_session():
    env = Env()
    try:
        board = BoardStore()
        platform = SessionPlatform()
        provider = ScriptedProvider(
            steps=[
                Step(text="first turn done"),
                Step(text="second turn done"),
            ]
        )
        ctx = RunContext(env=env, board=board, provider=provider, sessions=platform)
        info = await platform.create_session(
            "w", "start", toolset="plain", driver=drive_subagent, system="s", ctx=ctx
        )
        await platform.wait_all(timeout_s=10)
        result = await platform.send_message(
            info["id"], "go again", driver=drive_subagent, system="s", ctx=ctx
        )
        assert result["delivered"]
        await platform.wait_all(timeout_s=10)
        read = platform.read(info["id"], limit=50)
        user_msgs = [m for m in read["messages"] if m.get("role") == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[1]["content"] == "go again"
        assert platform.metadata(info["id"])["inbox"][0]["text"] == "go again"
    finally:
        env.cleanup()


async def test_session_read_unknown_hint():
    platform = SessionPlatform()
    with pytest.raises(KeyError, match="unknown session"):
        platform.read("ses-nope")


async def test_phrase_ids():
    from aibench.agentic.sessions import phrase_id

    ids = {phrase_id() for _ in range(50)}
    # 3-word space is 9600 wide — occasional collisions are expected; the
    # platform retries on collision (create_session loop)
    assert len(ids) >= 45
    for i in ids:
        assert i.count("-") == 2
