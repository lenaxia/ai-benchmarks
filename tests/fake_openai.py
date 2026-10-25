"""A scriptable OpenAI-compatible SSE server for tests (stdlib only)."""

from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAI:
    """Start with `with FakeOpenAI() as fake:`; enqueue responses, read `url`."""

    def __init__(self) -> None:
        self.script: deque[dict] = deque()
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # ---------- scripting API ----------
    def enqueue_text(self, text: str, usage: dict | None = None, pieces: int = 4):
        self.script.append({"kind": "text", "text": text, "usage": usage, "pieces": pieces})

    def enqueue_tool_call(self, name: str, arguments: str, usage: dict | None = None):
        self.script.append({"kind": "tool", "name": name, "arguments": arguments, "usage": usage})

    def enqueue_status(self, status: int, body: str):
        self.script.append({"kind": "status", "status": status, "body": body})

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    # ----------
    def __enter__(self) -> FakeOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _make_handler(self):  # noqa: PLR0915
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with outer._lock:
                    outer.requests.append(
                        {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
                    )
                    spec = (
                        outer.script.popleft()
                        if outer.script
                        else {"kind": "status", "status": 500, "body": "no script"}
                    )
                if spec["kind"] == "status":
                    payload = json.dumps({"error": spec["body"]}).encode()
                    self.send_response(spec["status"])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in outer._chunks(spec):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        return Handler

    @staticmethod
    def _chunks(spec: dict):
        rid = "chatcmpl-fake"
        base = {"id": rid, "object": "chat.completion.chunk", "model": "fake"}
        if spec["kind"] == "text":
            text = spec["text"]
            pieces = max(1, spec.get("pieces", 4))
            step = max(1, len(text) // pieces)
            parts = [text[i : i + step] for i in range(0, len(text), step)] or [text]
            for part in parts:
                yield {**base, "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]}
            yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        else:  # tool call streamed in fragments
            args = spec["arguments"]
            mid = max(1, len(args) // 2)
            yield {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_fake_1",
                                    "type": "function",
                                    "function": {"name": spec["name"], "arguments": args[:mid]},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[mid:]}}]},
                        "finish_reason": None,
                    }
                ],
            }
            yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        if spec.get("usage"):
            yield {**base, "choices": [], "usage": spec["usage"]}
