"""Tests for mount_ebs_volume SSH function."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aws_bootstrap.ssh import mount_ebs_volume


KEY_PATH = Path("/home/user/.ssh/id_ed25519.pub")


@pytest.fixture
def ssh_run():
    """Patch the ssh subprocess (exit 0); yields the mock."""
    with patch("aws_bootstrap.ssh.subprocess.run", return_value=MagicMock(returncode=0)) as run:
        yield run


def _mount(**kwargs):
    return mount_ebs_volume("1.2.3.4", "ubuntu", KEY_PATH, "vol-abc123", **kwargs)


def test_mount_ebs_volume_formats_a_new_volume(ssh_run):
    assert _mount(format_volume=True) is True
    ssh_run.assert_called_once()
    script = ssh_run.call_args[0][0][-1]
    assert "mkfs.ext4" in script
    assert "/data" in script
    assert "volabc123" in script  # stripped vol- hyphen


def test_mount_ebs_volume_never_formats_an_existing_volume(ssh_run):
    assert _mount(format_volume=False) is True
    ssh_run.assert_called_once()
    script = ssh_run.call_args[0][0][-1]
    assert "mkfs" not in script
    assert "/data" in script


def test_mount_ebs_volume_failure(ssh_run):
    """Non-zero exit code returns False."""
    ssh_run.return_value = MagicMock(returncode=1)
    assert _mount() is False


def test_mount_ebs_volume_custom_port(ssh_run):
    """Non-default port is passed as -p flag."""
    _mount(port=2222)
    cmd = ssh_run.call_args[0][0]
    assert cmd[cmd.index("-p") + 1] == "2222"


def test_mount_ebs_volume_custom_mount_point(ssh_run):
    """Custom mount point appears in the SSH script."""
    _mount(mount_point="/mnt/data")
    assert "/mnt/data" in ssh_run.call_args[0][0][-1]
