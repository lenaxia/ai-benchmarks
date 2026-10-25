"""Perf workload base classes and the request record."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aibench.providers.base import Provider
from aibench.types import ChatResponse


@dataclass
class RequestRecord:
    workload: str
    label: str
    target_prompt_tokens: int | None
    target_output_tokens: int | None
    response: ChatResponse
    derived: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        r = self.response
        out_tok = r.output_tokens()
        prompt_tok = r.prompt_tokens()
        row: dict[str, Any] = {
            "workload": self.workload,
            "label": self.label,
            "target_prompt_tokens": self.target_prompt_tokens,
            "target_output_tokens": self.target_output_tokens,
            "error": r.error,
            "finish_reason": r.finish_reason,
            "prompt_tokens": prompt_tok,
            "output_tokens": out_tok,
            "cached_tokens": r.usage.cached_tokens if r.usage else None,
            "timing": r.timing.to_dict(),
        }
        row.update(self.derived)
        return row


class Workload(ABC):
    name: str = "workload"

    @abstractmethod
    async def run(
        self, provider: Provider, repeats: int, warmup: int
    ) -> list[RequestRecord]: ...
