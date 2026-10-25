"""Hardware configuration models: CPU, RAM, GPUs — captured or manual."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CPUInfo:
    model: str = "unknown"
    sockets: int | None = None
    physical_cores: int | None = None
    threads: int | None = None
    mhz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "sockets": self.sockets,
            "physical_cores": self.physical_cores,
            "threads": self.threads,
            "mhz": self.mhz,
        }


@dataclass
class GPUInfo:
    vendor: str = "unknown"  # nvidia | amd | apple | other
    name: str = "unknown"
    vram_gb: float | None = None
    driver: str | None = None
    index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "driver": self.driver,
            "index": self.index,
            "extra": self.extra,
        }


@dataclass
class HardwareSnapshot:
    captured_at: str = ""
    source: str = "auto"  # auto | manual | mixed
    hostname: str = ""
    os: str = ""
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram_gb: float | None = None
    gpus: list[GPUInfo] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "source": self.source,
            "hostname": self.hostname,
            "os": self.os,
            "cpu": self.cpu.to_dict(),
            "ram_gb": self.ram_gb,
            "gpus": [g.to_dict() for g in self.gpus],
            "notes": self.notes,
        }

    def short(self) -> str:
        gpu = self.gpus[0].name if self.gpus else "no-gpu"
        if len(self.gpus) > 1:
            gpu = f"{len(self.gpus)}x {gpu}"
        cpu = self.cpu.model.split("(")[0].strip()[:40] or "unknown-cpu"
        ram = f"{self.ram_gb:.0f}GB" if self.ram_gb else "?GB"
        return f"{gpu} / {cpu} / {ram} RAM"
