import json

import pytest
from fake_openai import FakeOpenAI

from aibench.providers.openai_compat import OpenAICompatProvider


async def test_streams_text_and_usage():
    with FakeOpenAI() as fake:
        fake.enqueue_text(
            "hello world",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        )
        p = OpenAICompatProvider(base_url=fake.url, model="m", max_retries=0)
        resp = await p.chat(messages=[{"role": "user", "content": "hi"}], max_tokens=5)
        await p.close()
        assert resp.text == "hello world"
        assert resp.usage is not None
        assert resp.usage.cached_tokens == 8
        assert resp.usage.prompt_tokens == 10
        assert resp.timing.ttft_ms is not None
        assert resp.completion_tokens_observed >= 4  # streamed pieces


async def test_assembles_tool_call_fragments():
    with FakeOpenAI() as fake:
        fake.enqueue_tool_call("bash", json.dumps({"command": "echo hi"}))
        p = OpenAICompatProvider(base_url=fake.url, model="m", max_retries=0)
        resp = await p.chat(messages=[{"role": "user", "content": "go"}], tools=[])
        await p.close()
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "bash"
        assert json.loads(tc.arguments) == {"command": "echo hi"}
        assert resp.finish_reason == "tool_calls"


async def test_stream_options_fallback_on_400():
    with FakeOpenAI() as fake:
        fake.enqueue_status(400, "stream_options is not supported")
        fake.enqueue_text("ok")
        p = OpenAICompatProvider(base_url=fake.url, model="m", max_retries=0)
        resp = await p.chat(messages=[{"role": "user", "content": "hi"}])
        await p.close()
        assert resp.text == "ok"
        # second request went out without stream_options
        assert "stream_options" not in fake.requests[1]["body"]


async def test_retries_on_500_then_succeeds():
    with FakeOpenAI() as fake:
        fake.enqueue_status(500, "boom")
        fake.enqueue_status(500, "boom")
        fake.enqueue_text("finally")
        p = OpenAICompatProvider(base_url=fake.url, model="m", max_retries=3)
        resp = await p.chat(messages=[{"role": "user", "content": "hi"}])
        await p.close()
        assert resp.text == "finally"
        assert len(fake.requests) == 3


async def test_gives_up_after_max_retries():
    with FakeOpenAI() as fake:
        for _ in range(5):
            fake.enqueue_status(503, "down")
        p = OpenAICompatProvider(base_url=fake.url, model="m", max_retries=2)
        resp = await p.chat(messages=[{"role": "user", "content": "hi"}])
        await p.close()
        assert resp.error is not None
        assert "503" in resp.error or "down" in resp.error


@pytest.mark.parametrize("bad", [401, 404])
async def test_no_retry_on_client_errors(bad):
    with FakeOpenAI() as fake:
        fake.enqueue_status(bad, "nope")
        fake.enqueue_text("should not happen")
        p = OpenAICompatProvider(base_url=fake.url, model="m", max_retries=3)
        resp = await p.chat(messages=[{"role": "user", "content": "hi"}])
        await p.close()
        assert resp.error is not None
        assert len(fake.requests) == 1
