"""GPU architecture mapping and GPU info dataclass."""

from __future__ import annotations
from dataclasses import dataclass


_GPU_ARCHITECTURES: dict[str, str] = {
    "7.0": "Volta",
    "7.5": "Turing",
    "8.0": "Ampere",
    "8.6": "Ampere",
    "8.7": "Ampere",
    "8.9": "Ada Lovelace",
    "9.0": "Hopper",
    "10.0": "Blackwell",  # GB100 — B200 (p6-b200)
    "10.1": "Blackwell",
    "10.3": "Blackwell",  # GB300 "Blackwell Ultra" — B300 (p6-b300), sm_103 (CUDA 12.9+)
    "12.0": "Blackwell",  # GB20x — RTX PRO 4500/6000 Blackwell (g7, g7e)
    "12.1": "Blackwell",
}


@dataclass
class GpuInfo:
    """GPU information retrieved via nvidia-smi and nvcc."""

    driver_version: str
    cuda_driver_version: str  # max CUDA version supported by driver (from nvidia-smi)
    cuda_toolkit_version: str | None  # actual CUDA toolkit installed (from nvcc), None if unavailable
    gpu_name: str
    compute_capability: str
    architecture: str
