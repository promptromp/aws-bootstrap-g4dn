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
