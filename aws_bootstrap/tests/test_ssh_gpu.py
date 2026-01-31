"""Tests for GPU info queries via SSH (get_ssh_host_details, query_gpu_info)."""

from __future__ import annotations
import subprocess
from pathlib import Path
from unittest.mock import patch

from aws_bootstrap.ssh import (
    _GPU_ARCHITECTURES,
    GpuInfo,
    add_ssh_host,
    get_ssh_host_details,
    query_gpu_info,
)


KEY_PATH = Path("/home/user/.ssh/id_ed25519.pub")


# ---------------------------------------------------------------------------
# get_ssh_host_details
# ---------------------------------------------------------------------------


def test_get_ssh_host_details_found(tmp_path):
    """Parses HostName, User, IdentityFile from a managed SSH config block."""
    cfg = tmp_path / ".ssh" / "config"
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)

    details = get_ssh_host_details("i-abc123", config_path=cfg)
    assert details is not None
    assert details.hostname == "1.2.3.4"
    assert details.user == "ubuntu"
    assert details.identity_file == Path("/home/user/.ssh/id_ed25519")


def test_get_ssh_host_details_not_found(tmp_path):
    """Returns None when no managed block exists for the instance."""
    cfg = tmp_path / ".ssh" / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")

    assert get_ssh_host_details("i-missing", config_path=cfg) is None


def test_get_ssh_host_details_nonexistent_file(tmp_path):
    """Returns None when the SSH config file doesn't exist."""
    cfg = tmp_path / "no_such_file"
    assert get_ssh_host_details("i-abc123", config_path=cfg) is None


# ---------------------------------------------------------------------------
# query_gpu_info
# ---------------------------------------------------------------------------

NVIDIA_SMI_OUTPUT = "560.35.03, Tesla T4, 7.5\n12.8\n12.6\n"


@patch("aws_bootstrap.ssh.subprocess.run")
def test_query_gpu_info_success(mock_run):
    """Successful nvidia-smi + nvcc output returns a valid GpuInfo."""
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=NVIDIA_SMI_OUTPUT, stderr="")

    info = query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))
    assert info is not None
    assert isinstance(info, GpuInfo)
    assert info.driver_version == "560.35.03"
    assert info.cuda_driver_version == "12.8"
    assert info.cuda_toolkit_version == "12.6"
    assert info.gpu_name == "Tesla T4"
    assert info.compute_capability == "7.5"
    assert info.architecture == "Turing"


@patch("aws_bootstrap.ssh.subprocess.run")
def test_query_gpu_info_no_nvcc(mock_run):
    """When nvcc is unavailable, cuda_toolkit_version is None."""
    output = "560.35.03, Tesla T4, 7.5\n12.8\nN/A\n"
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr="")

    info = query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))
    assert info is not None
    assert info.cuda_driver_version == "12.8"
    assert info.cuda_toolkit_version is None


@patch("aws_bootstrap.ssh.subprocess.run")
def test_query_gpu_info_ssh_failure(mock_run):
    """Non-zero exit code returns None."""
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=255, stdout="", stderr="Connection refused")

    info = query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))
    assert info is None


@patch("aws_bootstrap.ssh.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=15))
def test_query_gpu_info_timeout(mock_run):
    """TimeoutExpired returns None."""
    info = query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))
    assert info is None


@patch("aws_bootstrap.ssh.subprocess.run")
def test_query_gpu_info_malformed_output(mock_run):
    """Garbage output returns None."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="not valid gpu output\n", stderr=""
    )

    info = query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))
    assert info is None


# ---------------------------------------------------------------------------
# GPU architecture mapping
# ---------------------------------------------------------------------------


def test_gpu_architecture_mapping():
    """Known compute capabilities map to correct architecture names."""
    assert _GPU_ARCHITECTURES["7.5"] == "Turing"
    assert _GPU_ARCHITECTURES["8.0"] == "Ampere"
    assert _GPU_ARCHITECTURES["8.6"] == "Ampere"
    assert _GPU_ARCHITECTURES["8.9"] == "Ada Lovelace"
    assert _GPU_ARCHITECTURES["9.0"] == "Hopper"
    assert _GPU_ARCHITECTURES["7.0"] == "Volta"


@patch("aws_bootstrap.ssh.subprocess.run")
def test_query_gpu_info_unknown_architecture(mock_run):
    """Unknown compute capability produces a fallback architecture string."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="550.00.00, Future GPU, 10.0\n13.0\n13.0\n", stderr=""
    )

    info = query_gpu_info("1.2.3.4", "ubuntu", Path("/home/user/.ssh/id_ed25519"))
    assert info is not None
    assert info.architecture == "Unknown (10.0)"
