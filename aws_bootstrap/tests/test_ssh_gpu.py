"""Tests for get_ssh_host_details (SSH config parsing)."""

from __future__ import annotations
from pathlib import Path

from aws_bootstrap.ssh import (
    add_ssh_host,
    get_ssh_host_details,
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
