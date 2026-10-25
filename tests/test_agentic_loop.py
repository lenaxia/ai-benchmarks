import json

import pytest

from aibench.agentic.board import BoardStore
from aibench.agentic.context import RunContext
from aibench.agentic.env import Env
from aibench.agentic.loop import run_agent_loop
from aibench.agentic.registry import build_registry
from aibench.testing import ScriptedProvider, Step


@pytest.fixture()
def world():
    env = Env()
    board = BoardStore()
    ctx = RunContext(env=env, board=board, provider=ScriptedProvider([]))
    yield ctx
    env.cleanup()


async def test_loop_executes_tools_and_finishes(world):
    world.provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "write", "arguments": {"path": "x.txt", "content": "hi"}}]),
            Step(tool_calls=[{"name": "bash", "arguments": {"command": "cat x.txt"}}]),
            Step(text="done: wrote and read x.txt"),
        ]
    )
    world.provider_ref = world.provider
    ctx = world
    ctx.provider = ctx.provider
    reg = build_registry("plain", ctx)
    res = await run_agent_loop(
        ctx.provider, system="s", prompt="do it", registry=reg, ctx=ctx, max_turns=10
    )
    assert res.stop_reason == "completed"
    assert res.final_text.startswith("done")
    assert ctx.env.read_file("x.txt") == "hi"
    assert ctx.metrics.by_tool == {"write": 1, "bash": 1}
    assert ctx.metrics.tool_calls == 2
    # transcript shape: system, user, assistant(tool_calls), tool, assistant(tool_calls), tool, assistant
    roles = [m["role"] for m in res.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert res.messages[2]["tool_calls"][0]["function"]["name"] == "write"


async def test_loop_handles_unknown_tool_and_bad_args(world):
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "teleport", "arguments": {"to": "moon"}}]),
            Step(tool_calls=[{"name": "write", "arguments": "{not json"}]),
            Step(text="recovered"),
        ]
    )
    ctx = world
    ctx.provider = provider
    reg = build_registry("plain", ctx)
    res = await run_agent_loop(
        provider, system="s", prompt="go", registry=reg, ctx=ctx, max_turns=10
    )
    assert res.stop_reason == "completed"
    assert ctx.metrics.arg_parse_failures == 1
    tool_msgs = [m for m in res.messages if m["role"] == "tool"]
    assert "unknown tool" in tool_msgs[0]["content"]
    assert "malformed JSON" in tool_msgs[1]["content"]


async def test_loop_max_turns(world):
    steps = [Step(tool_calls=[{"name": "bash", "arguments": {"command": "true"}}]) for _ in range(10)]
    provider = ScriptedProvider(steps=steps)
    ctx = world
    ctx.provider = provider
    reg = build_registry("plain", ctx)
    res = await run_agent_loop(
        provider, system="s", prompt="loop", registry=reg, ctx=ctx, max_turns=3
    )
    assert res.stop_reason == "max_turns"
    assert res.turns == 3


async def test_loop_tool_call_budget(world):
    steps = [Step(tool_calls=[{"name": "bash", "arguments": {"command": "true"}}]) for _ in range(50)]
    provider = ScriptedProvider(steps=steps)
    ctx = world
    ctx.provider = provider
    ctx.max_tool_calls = 4
    reg = build_registry("plain", ctx)
    res = await run_agent_loop(
        provider, system="s", prompt="loop", registry=reg, ctx=ctx, max_turns=100
    )
    assert res.stop_reason == "budget"
    assert ctx.metrics.tool_calls <= 5


async def test_subtask_tool_delegates(world):
    provider = ScriptedProvider(
        steps=[
            # parent delegates
            Step(tool_calls=[{
                "name": "subtask",
                "arguments": {"prompt": "read notes.txt and report its keyword", "agent": "explore"},
            }]),
            # child loop: reads file (readonly toolset) then reports
            Step(tool_calls=[{"name": "read", "arguments": {"path": "notes.txt"}}]),
            Step(text="keyword: MANGO"),
            # parent finishes with the report in context
            Step(text="subtask reported MANGO"),
        ]
    )
    ctx = world
    ctx.provider = provider
    ctx.env.write_file("notes.txt", "the keyword is MANGO\n")
    reg = build_registry("plain", ctx)
    res = await run_agent_loop(
        provider, system="s", prompt="delegate", registry=reg, ctx=ctx, max_turns=10
    )
    assert res.stop_reason == "completed"
    assert "MANGO" in res.final_text
    assert ctx.metrics.by_tool.get("subtask") == 1
    assert ctx.metrics.by_tool.get("read") == 1  # child rolled up into shared metrics
    # child tool call logged at depth 1
    assert any(c.depth == 1 and c.name == "read" for c in ctx.env.call_log)


async def test_board_tool_error_surfaces_as_json(world):
    provider = ScriptedProvider(
        steps=[
            Step(tool_calls=[{"name": "board_read", "arguments": {"task_id": "tsk-nope"}}]),
            Step(text="handled"),
        ]
    )
    ctx = world
    ctx.provider = provider
    reg = build_registry("safespace", ctx)
    res = await run_agent_loop(
        provider, system="s", prompt="go", registry=reg, ctx=ctx, max_turns=5
    )
    assert res.stop_reason == "completed"
    tool_out = [m for m in res.messages if m["role"] == "tool"][0]["content"]
    payload = json.loads(tool_out)
    assert payload["error"]["status"] == 404
    assert ctx.metrics.board_rejections == ["not_found"]


async def test_safespace_toolset_includes_plain_tools(world):
    ctx = world
    ctx.provider = ScriptedProvider([])
    reg = build_registry("safespace", ctx)
    names = reg.names()
    for expected in ("read", "write", "edit", "bash", "board_claim", "board_move_stage",
                     "session_create", "send_message", "get_datetime", "subtask"):
        assert expected in names, f"{expected} missing from {names}"
