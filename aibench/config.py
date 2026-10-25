"""Run configuration (YAML) with ${ENV_VAR} expansion."""

from __future__ import annotations

import os
import re
from typing import Any

import yaml
from pydantic import BaseModel, Field

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


class ProviderCfg(BaseModel):
    type: str = "openai"
    base_url: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str = ""
    timeout_s: float = 300.0
    max_retries: int = 3
    extra_headers: dict[str, str] = Field(default_factory=dict)
    stream_options: bool = True
    request_delay_s: float = 0.0


class HardwareCfg(BaseModel):
    capture: bool = True
    manual: dict[str, Any] | None = None  # {cpu, ram_gb, gpus, notes, hostname}


class DecodeCfg(BaseModel):
    enabled: bool = True
    context_levels: list[int] = Field(default_factory=lambda: [512, 2048, 8192, 32768])
    output_tokens: int = 256


class PrefillCfg(BaseModel):
    enabled: bool = True
    prompt_levels: list[int] = Field(default_factory=lambda: [1024, 8192, 32768, 131072])
    max_tokens: int = 32


class CacheCfg(BaseModel):
    enabled: bool = True
    prefix_tokens: int = 4096
    repeats: int = 5
    warmup: int = 1


class PerfCfg(BaseModel):
    enabled: bool = True
    repeats: int = 3
    warmup: int = 1
    concurrency: int = 1
    temperature: float | None = None
    chars_per_token: float = 4.0
    decode: DecodeCfg = Field(default_factory=DecodeCfg)
    prefill: PrefillCfg = Field(default_factory=PrefillCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)


class SubtaskCfg(BaseModel):
    max_depth: int = 1
    max_turns: int = 12


class AgenticCfg(BaseModel):
    enabled: bool = True
    scenarios: list[str] = Field(default_factory=lambda: ["all"])
    repeats: int = 1
    subtask: SubtaskCfg = Field(default_factory=SubtaskCfg)


class RunConfig(BaseModel):
    run_name: str = "run"
    notes: str = ""
    provider: ProviderCfg = Field(default_factory=ProviderCfg)
    hardware: HardwareCfg = Field(default_factory=HardwareCfg)
    perf: PerfCfg = Field(default_factory=PerfCfg)
    agentic: AgenticCfg = Field(default_factory=AgenticCfg)
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> RunConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.model_validate(_expand(raw))

    def to_public_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        # never persist secrets
        prov = d.get("provider", {})
        if prov.get("api_key"):
            prov["api_key"] = "***redacted***"
        return d
