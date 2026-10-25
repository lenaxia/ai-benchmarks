from aibench.hardware.detect import (
    parse_apple_gpus,
    parse_cpuinfo,
    parse_lscpu,
    parse_meminfo,
    parse_nvidia_smi,
    parse_rocminfo,
)

NVIDIA_CSV = """NVIDIA GeForce RTX 4090, 24564 MiB, 550.54.15
NVIDIA GeForce RTX 4090, 24564 MiB, 550.54.15
"""

LSCPU = """Architecture:        x86_64
CPU(s):              32
Model name:          AMD Ryzen 9 7950X 16-Core Processor
Thread(s) per core:  2
Core(s) per socket:  16
Socket(s):           1
CPU max MHz:         5879.000
"""

MEMINFO = """MemTotal:       65759940 kB
MemFree:        12345678 kB
"""

ROCM = """========================================
Marketing Name:            AMD Radeon PRO W7900
Memory Size:                48 GB
========================================
Marketing Name:            AMD Radeon PRO W7900
Memory Size:                48 GB
"""

APPLE = "Chipset Model: Apple M2 Max\nChipset Model: Apple M2 Max\n"


def test_nvidia_smi():
    gpus = parse_nvidia_smi(NVIDIA_CSV)
    assert len(gpus) == 2
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].name == "NVIDIA GeForce RTX 4090"
    assert abs(gpus[0].vram_gb - 23.98) < 0.1
    assert gpus[0].driver == "550.54.15"
    assert gpus[1].index == 1


def test_lscpu():
    cpu = parse_lscpu(LSCPU)
    assert cpu.model == "AMD Ryzen 9 7950X 16-Core Processor"
    assert cpu.sockets == 1
    assert cpu.physical_cores == 16
    assert cpu.threads == 32


def test_cpuinfo_fallback():
    cpu = parse_cpuinfo("model name\t: Intel i9\nprocessor\t: 0\nprocessor\t: 1\n")
    assert cpu.model == "Intel i9"
    assert cpu.threads == 2


def test_meminfo():
    assert abs(parse_meminfo(MEMINFO) - 62.7) < 0.5


def test_rocminfo():
    gpus = parse_rocminfo(ROCM)
    assert len(gpus) == 2
    assert gpus[0].vendor == "amd"
    assert gpus[0].vram_gb == 48.0


def test_apple():
    gpus = parse_apple_gpus(APPLE)
    assert len(gpus) == 2
    assert gpus[0].name == "Apple M2 Max"
