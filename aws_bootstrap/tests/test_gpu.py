"""Tests for GPU info queries via SSH (query_gpu_info, GPU architecture mapping)."""

from __future__ import annotations
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aws_bootstrap.gpu import _GPU_ARCHITECTURES, GpuInfo
from aws_bootstrap.ssh import query_gpu_info


def _query(stdout: str = "", returncode: int = 0, stderr: str = "", error: Exception | None = None):
    """Run query_gpu_info against a faked ssh returning ``stdout`` (or raising ``error``)."""
    completed = subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
    with patch("aws_bootstrap.ssh.subprocess.run", return_value=completed, side_effect=error):
        return query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))


def test_query_gpu_info_success():
    """Successful nvidia-smi + nvcc output returns a valid GpuInfo."""
    info = _query("560.35.03, Tesla T4, 7.5\n12.8\n12.6\n")
    assert info == GpuInfo(
        driver_version="560.35.03",
        cuda_driver_version="12.8",
        cuda_toolkit_version="12.6",
        gpu_name="Tesla T4",
        compute_capability="7.5",
        architecture="Turing",
    )


def test_query_gpu_info_no_nvcc():
    """When nvcc is unavailable, cuda_toolkit_version is None."""
    info = _query("560.35.03, Tesla T4, 7.5\n12.8\nN/A\n")
    assert info is not None
    assert info.cuda_driver_version == "12.8"
    assert info.cuda_toolkit_version is None


def test_query_gpu_info_unknown_architecture():
    """Unknown compute capability produces a fallback architecture string."""
    info = _query("550.00.00, Future GPU, 99.0\n13.0\n13.0\n")
    assert info is not None
    assert info.architecture == "Unknown (99.0)"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"returncode": 255, "stderr": "Connection refused"}, id="ssh-failure"),
        pytest.param({"error": subprocess.TimeoutExpired(cmd="ssh", timeout=15)}, id="timeout"),
        pytest.param({"stdout": "not valid gpu output\n"}, id="malformed-output"),
    ],
)
def test_query_gpu_info_returns_none_on_failure(kwargs):
    assert _query(**kwargs) is None


@pytest.mark.parametrize(
    "capability,architecture",
    [
        ("7.0", "Volta"),
        ("7.5", "Turing"),
        ("8.0", "Ampere"),
        ("8.6", "Ampere"),
        ("8.9", "Ada Lovelace"),
        ("9.0", "Hopper"),
        # Blackwell datacenter parts backing the current P6 families:
        # sm_100 = GB100/B200 (p6-b200), sm_103 = GB300 "Blackwell Ultra"/B300 (p6-b300).
        ("10.0", "Blackwell"),
        ("10.3", "Blackwell"),
        # sm_120 = GB20x, the RTX PRO 4500/6000 Blackwell parts behind g7/g7e.
        ("12.0", "Blackwell"),
    ],
)
def test_gpu_architecture_mapping(capability, architecture):
    assert _GPU_ARCHITECTURES[capability] == architecture
