"""Tests for mount_ebs_volume SSH function."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

from aws_bootstrap.ssh import mount_ebs_volume


KEY_PATH = Path("/home/user/.ssh/id_ed25519.pub")


@patch("aws_bootstrap.ssh.subprocess.run")
def test_mount_ebs_volume_success_format(mock_run):
    """New volume: SSH command includes mkfs."""
    mock_run.return_value = MagicMock(returncode=0)

    result = mount_ebs_volume("1.2.3.4", "ubuntu", KEY_PATH, "vol-abc123", format_volume=True)

    assert result is True
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    script = cmd[-1]
    assert "mkfs.ext4" in script
    assert "/data" in script
    assert "volabc123" in script  # stripped vol- hyphen


@patch("aws_bootstrap.ssh.subprocess.run")
def test_mount_ebs_volume_success_no_format(mock_run):
    """Existing volume: SSH command skips mkfs."""
    mock_run.return_value = MagicMock(returncode=0)

    result = mount_ebs_volume("1.2.3.4", "ubuntu", KEY_PATH, "vol-abc123", format_volume=False)

    assert result is True
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    script = cmd[-1]
    assert "mkfs" not in script
    assert "/data" in script


@patch("aws_bootstrap.ssh.subprocess.run")
def test_mount_ebs_volume_failure(mock_run):
    """Non-zero exit code returns False."""
    mock_run.return_value = MagicMock(returncode=1)

    result = mount_ebs_volume("1.2.3.4", "ubuntu", KEY_PATH, "vol-abc123")

    assert result is False


@patch("aws_bootstrap.ssh.subprocess.run")
def test_mount_ebs_volume_custom_port(mock_run):
    """Non-default port is passed as -p flag."""
    mock_run.return_value = MagicMock(returncode=0)

    mount_ebs_volume("1.2.3.4", "ubuntu", KEY_PATH, "vol-abc123", port=2222)

    cmd = mock_run.call_args[0][0]
    assert "-p" in cmd
    port_idx = cmd.index("-p")
    assert cmd[port_idx + 1] == "2222"


@patch("aws_bootstrap.ssh.subprocess.run")
def test_mount_ebs_volume_custom_mount_point(mock_run):
    """Custom mount point appears in the SSH script."""
    mock_run.return_value = MagicMock(returncode=0)

    mount_ebs_volume("1.2.3.4", "ubuntu", KEY_PATH, "vol-abc123", mount_point="/mnt/data")

    cmd = mock_run.call_args[0][0]
    script = cmd[-1]
    assert "/mnt/data" in script
