"""Provider abstraction. OpenAI-compatible endpoints are the primary target;
other wire protocols can be added by implementing :class:`Provider` and
registering it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aibench.types import ChatResponse, Message, ToolSchema

_PROVIDERS: dict[str, type[Provider]] = {}


def register_provider(type_name: str):
    def deco(cls: type[Provider]) -> type[Provider]:
        _PROVIDERS[type_name] = cls
        return cls

    return deco


def create_provider(type_name: str, **kwargs: Any) -> Provider:
    if type_name not in _PROVIDERS:
        raise KeyError(
            f"unknown provider type {type_name!r}; registered: {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[type_name](**kwargs)


class Provider(ABC):
    """A chat-completions backend.

    Implementations must stream responses (timing capture depends on it) and
    return a normalized :class:`ChatResponse`.
    """

    name: str = "provider"

    @abstractmethod
    async def chat(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        tool_choice: Any | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResponse: ...

    async def close(self) -> None: ...  # noqa: B027


def tool_spec(name: str, description: str, parameters: dict[str, Any]) -> ToolSchema:
    """Build an OpenAI-format function-tool schema."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }
