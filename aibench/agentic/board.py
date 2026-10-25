"""In-memory mock of the llmsafespaces Task Board.

Faithfully reproduces the semantics observed in production sessions:

- claim tokens + leases (default 15 min) — every in-flight write needs the token
- heartbeats require *new* progress evidence: strictly-monotonic watermark,
  future stamps rejected (400 ``future_stamp``)
- ``move_stage(done)`` is refused (409 ``criteria_outstanding``) until every
  acceptance criterion is marked met and an evidenceUrl exists
- stale/incorrect tokens yield 409 ``claim_token`` — the stand-down signal
- every mutation appends to the task audit log (dependency-order checks)
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

STAGES = ("backlog", "in_progress", "in_review", "done", "pending_input")


class BoardError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def payload(self) -> dict[str, Any]:
        return {"error": {"status": self.status, "code": self.code, "message": self.message}}


@dataclass
class Claim:
    session_id: str
    token: str
    expires_at: float
    watermark: float | None = None


@dataclass
class TaskRow:
    id: str
    epic_id: str
    title: str
    description: str = ""
    criteria: list[dict[str, Any]] = field(default_factory=list)  # {text, met}
    stage: str = "backlog"
    depends_on: list[str] = field(default_factory=list)
    claim: Claim | None = None
    attempts: int = 0
    evidence_url: str | None = None
    audit: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self, include_audit: bool = True) -> dict[str, Any]:
        d = {
            "id": self.id,
            "epic_id": self.epic_id,
            "title": self.title,
            "description": self.description,
            "criteria": list(self.criteria),
            "stage": self.stage,
            "depends_on": list(self.depends_on),
            "attempts": self.attempts,
            "evidence_url": self.evidence_url,
            "claim": {
                "session_id": self.claim.session_id,
                "lease_expires_at": self.claim.expires_at,
            }
            if self.claim
            else None,
        }
        if include_audit:
            d["audit"] = self.audit[-50:]
        return d


@dataclass
class EpicRow:
    id: str
    project_id: str
    title: str
    outcome: str = ""
    active: bool = False
    task_ids: list[str] = field(default_factory=list)


@dataclass
class ProjectRow:
    id: str
    name: str
    worker_slots: int = 2
    epic_ids: list[str] = field(default_factory=list)


def _short_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(3)}"


class _SeqIds:
    """Deterministic ids for tests (tsk-0001, tsk-0002, ...)."""

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        return f"{prefix}-{self.counters[prefix]:04d}"


class BoardStore:
    def __init__(
        self,
        now: Callable[[], float] = time.time,
        lease_ttl_s: float = 900.0,
        require_evidence_for_review: bool = True,
        future_grace_s: float = 5.0,
        deterministic_ids: bool = False,
    ):
        self.now = now
        self.lease_ttl_s = lease_ttl_s
        self.require_evidence_for_review = require_evidence_for_review
        self.future_grace_s = future_grace_s
        self._id_factory: Callable[[str], str] = _SeqIds() if deterministic_ids else _short_id
        self.projects: dict[str, ProjectRow] = {}
        self.epics: dict[str, EpicRow] = {}
        self.tasks: dict[str, TaskRow] = {}

    @property
    def _id(self) -> Callable[[str], str]:
        return self._id_factory

    # ------------- audit -------------
    def _audit(self, task: TaskRow, event: str, detail: str = "") -> None:
        task.audit.append({"ts": self.now(), "event": event, "detail": detail})

    # ------------- structure -------------
    def create_project(self, name: str, worker_slots: int = 2) -> dict[str, Any]:
        pid = self._id("prj")
        self.projects[pid] = ProjectRow(id=pid, name=name, worker_slots=worker_slots)
        return {"id": pid, "name": name, "worker_slots": worker_slots}

    def create_epic(self, project_id: str, title: str, outcome: str = "") -> dict[str, Any]:
        if project_id not in self.projects:
            raise BoardError(404, "not_found", f"project {project_id} not found")
        eid = self._id("epc")
        self.epics[eid] = EpicRow(id=eid, project_id=project_id, title=title, outcome=outcome)
        self.projects[project_id].epic_ids.append(eid)
        return {"id": eid, "title": title, "outcome": outcome}

    def create_task(
        self,
        epic_id: str,
        title: str,
        description: str = "",
        criteria: list[str] | None = None,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        if epic_id not in self.epics:
            raise BoardError(404, "not_found", f"epic {epic_id} not found")
        deps = depends_on or []
        for dep in deps:
            if dep not in self.tasks:
                raise BoardError(404, "not_found", f"dependency {dep} not found")
        # cycle guard
        seen: set[str] = set()
        stack = list(deps)
        while stack:
            cur = stack.pop()
            if cur == epic_id or (epic_id.startswith("epc") and cur == epic_id):
                pass
            if cur in seen:
                raise BoardError(409, "cycle", "dependency cycle detected")
            seen.add(cur)
            stack.extend(self.tasks[cur].depends_on if cur in self.tasks else [])
        tid = self._id("tsk")
        row = TaskRow(
            id=tid,
            epic_id=epic_id,
            title=title,
            description=description,
            criteria=[{"text": c, "met": False} for c in (criteria or [])],
            depends_on=deps,
        )
        self.tasks[tid] = row
        self.epics[epic_id].task_ids.append(tid)
        self._audit(row, "created", title)
        return row.to_dict(include_audit=False)

    # ------------- claim / lease -------------
    def _deps_ready(self, task: TaskRow) -> bool:
        return all(self.tasks[d].stage == "done" for d in task.depends_on if d in self.tasks)

    def claim(self, task_id: str, session_id: str) -> dict[str, Any]:
        task = self._get(task_id)
        t = self.now()
        if task.claim and task.claim.expires_at > t:
            if task.claim.session_id != session_id:
                raise BoardError(
                    409, "claim_active", f"task already claimed by {task.claim.session_id}"
                )
            # same session re-claim: return existing charter
        else:
            if task.stage not in ("backlog", "in_progress"):
                raise BoardError(
                    409, "stage_conflict", f"task in stage {task.stage} is not claimable"
                )
            if not self._deps_ready(task):
                blocked = [d for d in task.depends_on if self.tasks[d].stage != "done"]
                raise BoardError(
                    409, "blocked", f"dependencies not done: {', '.join(blocked)}"
                )
            task.attempts += 1
            task.claim = Claim(
                session_id=session_id,
                token=self._id("clm"),
                expires_at=t + self.lease_ttl_s,
            )
            if task.stage == "backlog":
                task.stage = "in_progress"
            self._audit(task, "claimed", f"attempt {task.attempts} by {session_id}")
        task.claim.expires_at = t + self.lease_ttl_s
        return {
            "task": task.to_dict(include_audit=False),
            "claim_token": task.claim.token,
            "lease_expires_at": task.claim.expires_at,
            "charter": task.description,
            "criteria": task.criteria,
        }

    def heartbeat(self, task_id: str, claim_token: str, observed_at: float) -> dict[str, Any]:
        task = self._get(task_id)
        t = self.now()
        self._check_claim(task, claim_token)
        if observed_at > t + self.future_grace_s:
            raise BoardError(
                400, "future_stamp", "future stamps are not progress evidence"
            )
        if task.claim is not None and task.claim.watermark is not None and observed_at <= task.claim.watermark:
            raise BoardError(
                409,
                "not_progress",
                "observedProgressAt must advance past the stored watermark",
            )
        assert task.claim is not None  # checked by _check_claim
        task.claim.watermark = observed_at
        task.claim.expires_at = t + self.lease_ttl_s
        self._audit(task, "heartbeat", f"watermark={observed_at:.1f}")
        return {"ok": True, "lease_expires_at": task.claim.expires_at}

    def _check_claim(self, task: TaskRow, claim_token: str) -> None:
        t = self.now()
        if not task.claim:
            raise BoardError(409, "claim_token", "no active claim on this task")
        if task.claim.token != claim_token:
            raise BoardError(
                409,
                "claim_token",
                "claim token mismatch — lease may have lapsed and been reclaimed; "
                "STOP working this task and re-read it",
            )
        if task.claim.expires_at <= t:
            raise BoardError(
                409, "claim_token", "lease lapsed — STOP working this task and re-read it"
            )

    # ------------- stage machine -------------
    def move_stage(
        self,
        task_id: str,
        to_stage: str,
        claim_token: str | None = None,
        evidence_url: str | None = None,
    ) -> dict[str, Any]:
        task = self._get(task_id)
        if to_stage not in STAGES:
            raise BoardError(400, "bad_stage", f"unknown stage {to_stage!r}")
        frm = task.stage
        if to_stage == frm:
            raise BoardError(409, "no_op", f"task already in {to_stage}")

        if to_stage == "in_review":
            if frm != "in_progress":
                raise BoardError(
                    409, "stage_conflict", f"in_review requires in_progress, not {frm}"
                )
            if task.claim:
                if not claim_token:
                    raise BoardError(
                        409, "claim_token", "claim token required for in-flight stage moves"
                    )
                self._check_claim(task, claim_token)
            if self.require_evidence_for_review and not evidence_url:
                raise BoardError(
                    400,
                    "evidence_required",
                    "evidenceUrl is required for in_review",
                )
            task.evidence_url = evidence_url
            task.stage = "in_review"
            self._audit(task, "stage:in_review", evidence_url or "")

        elif to_stage == "done":
            if frm != "in_review":
                raise BoardError(
                    409, "stage_conflict", f"done requires in_review, not {frm}"
                )
            unmet = [c["text"] for c in task.criteria if not c.get("met")]
            if unmet:
                raise BoardError(
                    409,
                    "criteria_outstanding",
                    "refused: acceptance criteria not marked met: " + "; ".join(unmet),
                )
            if not task.evidence_url and not evidence_url:
                raise BoardError(400, "evidence_required", "evidenceUrl is required for done")
            if evidence_url:
                task.evidence_url = evidence_url
            task.stage = "done"
            task.claim = None
            self._audit(task, "stage:done", task.evidence_url or "")

        elif to_stage == "in_progress":
            if frm not in ("backlog", "pending_input"):
                raise BoardError(409, "stage_conflict", f"in_progress not reachable from {frm}")
            if not self._deps_ready(task):
                blocked = [d for d in task.depends_on if self.tasks[d].stage != "done"]
                raise BoardError(409, "blocked", f"dependencies not done: {', '.join(blocked)}")
            task.stage = "in_progress"
            self._audit(task, "stage:in_progress", "")

        elif to_stage == "pending_input":
            if frm not in ("in_progress", "in_review"):
                raise BoardError(409, "stage_conflict", f"pending_input not reachable from {frm}")
            task.stage = "pending_input"
            self._audit(task, "stage:pending_input", "")

        return task.to_dict()

    # ------------- updates -------------
    def update_task(
        self,
        task_id: str,
        description: str | None = None,
        set_criteria_met: int | None = None,
        met: bool = True,
        title: str | None = None,
    ) -> dict[str, Any]:
        task = self._get(task_id)
        if description is not None:
            task.description = description
            self._audit(task, "description", description[:200])
        if title is not None:
            task.title = title
            self._audit(task, "title", title)
        if set_criteria_met is not None:
            if not task.criteria:
                raise BoardError(400, "no_criteria", "task has no acceptance criteria")
            if not 0 <= set_criteria_met < len(task.criteria):
                raise BoardError(
                    400,
                    "bad_index",
                    f"criteria index {set_criteria_met} out of range 0..{len(task.criteria) - 1}",
                )
            task.criteria[set_criteria_met]["met"] = bool(met)
            self._audit(
                task,
                f"criterion[{set_criteria_met}]={'met' if met else 'unmet'}",
                task.criteria[set_criteria_met]["text"][:200],
            )
        return task.to_dict(include_audit=False)

    def read(self, task_id: str) -> dict[str, Any]:
        task = self._get(task_id)
        blocked_by = [
            {"id": d, "stage": self.tasks[d].stage} for d in task.depends_on if d in self.tasks
        ]
        d = task.to_dict()
        d["blocked_by"] = blocked_by
        return d

    def _get(self, task_id: str) -> TaskRow:
        if task_id not in self.tasks:
            raise BoardError(404, "not_found", f"task {task_id} not found")
        return self.tasks[task_id]

    # ------------- whole-board view (for checks/reports) -------------
    def state(self) -> dict[str, Any]:
        return {
            "projects": [
                {**{"id": p.id, "name": p.name, "worker_slots": p.worker_slots},
                 "epics": [
                     {"id": e.id, "title": e.title, "active": e.active}
                     for e in self.epics.values()
                     if e.project_id == p.id
                 ]}
                for p in self.projects.values()
            ],
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
        }
