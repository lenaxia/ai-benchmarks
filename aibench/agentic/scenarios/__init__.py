from aibench.agentic.scenarios.base import CheckResult, Scenario, SetupBundle
from aibench.agentic.scenarios.board_scenarios import BOARD_SCENARIOS
from aibench.agentic.scenarios.plain_scenarios import PLAIN_SCENARIOS

ALL_SCENARIOS: list[Scenario] = [*PLAIN_SCENARIOS, *BOARD_SCENARIOS]

SCENARIOS: dict[str, Scenario] = {s.name: s for s in ALL_SCENARIOS}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {name!r}; available: {', '.join(sorted(SCENARIOS))}"
        )
    return SCENARIOS[name]


def list_scenarios() -> list[dict]:
    return [
        {
            "name": s.name,
            "title": s.title,
            "toolset": s.toolset,
            "checks": [name for name, _ in s.checks],
            "max_turns": s.max_turns,
            "max_tool_calls": s.max_tool_calls,
        }
        for s in ALL_SCENARIOS
    ]

__all__ = [
    "Scenario",
    "SetupBundle",
    "CheckResult",
    "ALL_SCENARIOS",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
]
