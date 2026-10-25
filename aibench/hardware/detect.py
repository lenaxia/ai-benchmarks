"""Hardware detection: GPU (nvidia-smi / rocminfo / Apple), CPU, RAM.

Every probe degrades gracefully — a benchmark laptop, a headless CI box and a
8xH100 node should all produce a snapshot, with ``unknown`` where detection
is unavailable. Manual overrides in the run config win over detection.
"""

from __future__ import annotations

import asyncio
import platform
import re
import socket
from datetime import UTC, datetime
from typing import Any

from aibench.hardware.models import CPUInfo, GPUInfo, HardwareSnapshot


async def _run(cmd: list[str]) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        return out.decode(errors="replace")
    except (TimeoutError, OSError, FileNotFoundError):
        return None


def parse_lscpu(text: str) -> CPUInfo:
    info = CPUInfo()
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
    info.model = fields.get("Model name") or "unknown"
    sockets = fields.get("Socket(s)")
    cores = fields.get("Core(s) per socket")
    threads = fields.get("CPU(s)")
    mhz = fields.get("CPU max MHz") or fields.get("CPU MHz")
    import contextlib

    with contextlib.suppress(ValueError):
        info.sockets = int(sockets) if sockets else None
    with contextlib.suppress(ValueError):
        per_socket = int(cores) if cores else None
        if per_socket and info.sockets:
            info.physical_cores = per_socket * info.sockets
    with contextlib.suppress(ValueError):
        info.threads = int(threads) if threads else None
    with contextlib.suppress(ValueError):
        info.mhz = float(mhz) if mhz else None
    return info


def parse_cpuinfo(text: str) -> CPUInfo:
    info = CPUInfo()
    models: list[str] = []
    threads = 0
    for line in text.splitlines():
        if line.startswith("model name") and ":" in line:
            models.append(line.partition(":")[2].strip())
        elif line.startswith("processor"):
            threads += 1
    info.model = models[0] if models else "unknown"
    info.threads = threads or None
    return info


def parse_meminfo(text: str) -> float | None:
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return round(kb / (1024 * 1024), 2)
    return None


def parse_nvidia_smi(csv: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for i, line in enumerate(x.strip() for x in csv.splitlines() if x.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        vram = None
        m = re.search(r"([\d.]+)\s*(MiB|GiB|B)", parts[1])
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            vram = (
                val / 1024 if unit == "MiB" else val if unit == "GiB" else val / (1024**3)
            )
            vram = round(vram, 2)
        driver = parts[2] if len(parts) > 2 and parts[2] != "[Not Supported]" else None
        gpus.append(GPUInfo(vendor="nvidia", name=name, vram_gb=vram, driver=driver, index=i))
    return gpus


def parse_rocminfo(text: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    blocks = re.split(r"========================================", text)
    for block in blocks:
        m = re.search(r"Marketing Name:\s*(.+)", block)
        if not m:
            continue
        name = m.group(1).strip()
        if not name or name.lower() == "device":
            continue
        vram = None
        vm = re.search(r"Memory Size:\s*([\d.]+)\s*(\w+)", block)
        if vm:
            val = float(vm.group(1))
            unit = vm.group(2).lower()
            vram = val / 1024 if unit in ("mb", "mib") else round(val, 2) if unit in ("gb", "gib") else None
        gpus.append(GPUInfo(vendor="amd", name=name, vram_gb=vram, index=len(gpus)))
        if len(gpus) >= 16:
            break
    return gpus


def parse_apple_gpus(text: str) -> list[GPUInfo]:
    gpus = []
    for m in re.finditer(r"Chipset Model:\s*(.+)", text):
        gpus.append(GPUInfo(vendor="apple", name=m.group(1).strip(), index=len(gpus)))
    return gpus


async def detect_cpu() -> CPUInfo:
    out = await _run(["lscpu"])
    if out:
        info = parse_lscpu(out)
        if info.model != "unknown":
            return info
    try:
        with open("/proc/cpuinfo") as f:
            return parse_cpuinfo(f.read())
    except OSError:
        return CPUInfo(model=platform.processor() or platform.machine() or "unknown")


async def detect_ram() -> float | None:
    try:
        with open("/proc/meminfo") as f:
            return parse_meminfo(f.read())
    except OSError:
        pass
    if platform.system() == "Darwin":
        out = await _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return round(int(out.strip()) / (1024**3), 2)
            except ValueError:
                pass
    return None


async def detect_gpus() -> list[GPUInfo]:
    out = await _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if out:
        gpus = parse_nvidia_smi(out)
        if gpus:
            return gpus
    out = await _run(["rocminfo"])
    if out:
        gpus = parse_rocminfo(out)
        if gpus:
            return gpus
    if platform.system() == "Darwin":
        out = await _run(["system_profiler", "SPDisplaysDataType"])
        if out:
            return parse_apple_gpus(out)
    return []


async def detect_hardware(manual: dict[str, Any] | None = None) -> HardwareSnapshot:
    """Capture a snapshot; keys in ``manual`` override detected values."""
    cpu, ram, gpus = await asyncio.gather(detect_cpu(), detect_ram(), detect_gpus())
    snap = HardwareSnapshot(
        captured_at=datetime.now(UTC).isoformat(),
        source="auto",
        hostname=socket.gethostname(),
        os=f"{platform.system()} {platform.release()}",
        cpu=cpu,
        ram_gb=ram,
        gpus=gpus,
    )
    if manual:
        snap.source = "manual" if manual.get("replace_all") else "mixed"
        if manual.get("hostname"):
            snap.hostname = manual["hostname"]
        if manual.get("cpu"):
            snap.cpu = CPUInfo(**{**snap.cpu.to_dict(), **manual["cpu"]})
        if manual.get("ram_gb") is not None:
            snap.ram_gb = manual["ram_gb"]
        if manual.get("gpus") is not None:
            snap.gpus = [GPUInfo(**g) for g in manual["gpus"]]
        if manual.get("notes"):
            snap.notes = manual["notes"]
    return snap


def detect_hardware_sync(manual: dict[str, Any] | None = None) -> HardwareSnapshot:
    return asyncio.run(detect_hardware(manual))
