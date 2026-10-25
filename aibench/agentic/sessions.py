"""Mock llmsafespaces session platform: fire-and-forget worker sessions.

session_create spawns a *real* nested agent loop (driven by the benchmarked
model) in the background; send_message queues a next turn; session_read /
session_list / session_metadata inspect state. Semantics mirror the real
platform closely enough that orchestration behaviour is measurable:
- create returns immediately with a session id
- messages queue while a session is busy and are delivered when the current
  turn ends
- worker sessions share the parent's workspace and board
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# 3-word phrase ids in the real platform's style ("valued-titanium-hauls")
_ADJ = [
    "amber", "bold", "calm", "deft", "eager", "faint", "glassy", "handy",
    "ivory", "jolly", "keen", "lazy", "misty", "noble", "opal", "pristine",
    "quiet", "rival", "slick", "tender", "valued", "witty", "zesty", "mellow",
]
_NOUN = [
    "panda", "canyon", "nebula", "orchid", "falcon", "titanium", "porcupine",
    "diagram", "vestibule", "tram", "sitar", "kelp", "monastery", "squeegee",
    "milestone", "carousel", "lantern", "songbird", "bloom", "canal",
]
_VERB = [
    "hauls", "strolls", "recants", "invokes", "puzzles", "supports", "weeds",
    "loses", "tramps", "seizes", "snorts", "masks", "types", "rushes",
    "cheers", "waits", "endures", "covers", "refrains", "indicates",
]


def phrase_id() -> str:
    return f"{random.choice(_ADJ)}-{random.choice(_NOUN)}-{random.choice(_VERB)}"


LoopDriver = Callable[..., Awaitable["tuple[str, list[dict]]"]]


@dataclass
class SessionRecord:
    id: str
    title: str
    status: str = "starting"  # starting|busy|idle|done|error
    toolset: str = "plain"
    created_at: float = field(default_factory=time.time)
    messages: list[dict[str, Any]] = field(default_factory=list)
    inbox: list[dict[str, Any]] = field(default_factory=list)  # {from, text, ts}
    errors: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "toolset": self.toolset,
            "created_at": self.created_at,
            "message_count": len(self.messages),
        }


class SessionPlatform:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionRecord] = {}
        self._queues: dict[str, asyncio.Queue[str]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # -------- called by tools --------
    async def create_session(
        self,
        title: str,
        prompt: str,
        *,
        toolset: str = "plain",
        driver: LoopDriver | None = None,
        system: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        sid = phrase_id()
        while sid in self.sessions:
            sid = phrase_id()
        rec = SessionRecord(id=sid, title=title, toolset=toolset, status="starting")
        self.sessions[sid] = rec
        self._queues[sid] = asyncio.Queue()
        await self._enqueue(sid, prompt, driver=driver, system=system, ctx=ctx)
        return rec.describe()

    async def send_message(
        self,
        session_id: str,
        text: str,
        *,
        from_id: str = "orchestrator",
        driver: LoopDriver | None = None,
        system: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        rec = self._get(session_id)
        rec.inbox.append({"from": from_id, "text": text, "ts": time.time()})
        running = self._tasks.get(session_id) and not self._tasks[session_id].done()
        await self._enqueue(session_id, text, driver=driver, system=system, ctx=ctx)
        return {
            "delivered": True,
            "status": "delivering_after_current_turn" if running else "delivered_now",
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        return [s.describe() for s in self.sessions.values()]

    def read(self, session_id: str, limit: int = 20) -> dict[str, Any]:
        rec = self._get(session_id)
        msgs = rec.messages[-limit:]
        return {"session_id": rec.id, "title": rec.title, "status": rec.status,
                "messages": msgs}

    def metadata(self, session_id: str) -> dict[str, Any]:
        rec = self._get(session_id)
        d = rec.describe()
        d["busy"] = bool(self._tasks.get(session_id) and not self._tasks[session_id].done())
        d["inbox"] = rec.inbox
        d["errors"] = rec.errors
        return d

    # -------- internals --------
    async def _enqueue(
        self,
        sid: str,
        prompt: str,
        *,
        driver: LoopDriver | None,
        system: str | None,
        ctx: Any,
    ) -> None:
        q = self._queues.setdefault(sid, asyncio.Queue())
        await q.put(prompt)
        running = self._tasks.get(sid) and not self._tasks[sid].done()
        if not running and driver is not None:
            self._tasks[sid] = asyncio.create_task(
                self._drive(sid, driver, system, ctx)
            )

    async def _drive(self, sid: str, driver: LoopDriver, system: str | None, ctx: Any) -> None:

        rec = self.sessions[sid]
        q = self._queues[sid]
        rec.status = "busy"
        try:
            while not q.empty():
                prompt = await q.get()
                history = [m for m in rec.messages if m.get("role") != "system_note"]
                child = ctx.child(session_id=sid, toolset=rec.toolset)
                try:
                    final_text, messages = await driver(
                        prompt=prompt, system=system, ctx=child, history=history
                    )
                except Exception as e:  # noqa: BLE001
                    rec.errors.append(f"{type(e).__name__}: {e}")
                    rec.messages.append(
                        {"role": "assistant", "content": f"[session error: {e}]",
                         "ts": time.time()}
                    )
                    continue
                rec.messages = messages
                rec.messages.append({"role": "system_note", "content": "turn complete", "ts": time.time()})
        finally:
            rec.status = "done" if not rec.errors else "error"

    def _get(self, sid: str) -> SessionRecord:
        if sid not in self.sessions:
            # near-miss helper, real API style
            candidates = [s for s in self.sessions if s.startswith(sid[:6])]
            hint = f" did you mean: {candidates}" if candidates else ""
            raise KeyError(f"unknown session {sid!r}.{hint}")
        return self.sessions[sid]

    # -------- for scenario setup --------
    def seed_session(
        self,
        title: str,
        messages: list[dict[str, Any]],
        status: str = "done",
        sid: str | None = None,
    ) -> SessionRecord:
        sid = sid or phrase_id()
        while sid in self.sessions:
            sid = phrase_id()
        rec = SessionRecord(id=sid, title=title, status=status, messages=messages)
        self.sessions[sid] = rec
        return rec

    async def wait_all(self, timeout_s: float = 120.0) -> None:
        tasks = [t for t in self._tasks.values() if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout_s)
