"""Tests for generic remote-exec SSH primitives (run_on_host, scp_to_host)."""

from __future__ import annotations
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aws_bootstrap.constants import SSH_CONNECT_TIMEOUT
from aws_bootstrap.ssh import _port_opts, _ssh_opts, run_on_host, scp_to_host


KEY = Path("/home/user/.ssh/id_ed25519.pub")


@pytest.fixture
def ssh_run():
    """Patch the ssh/scp subprocess (exit 0, no output); yields the mock."""
    with patch("aws_bootstrap.ssh.subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")) as run:
        yield run


@pytest.fixture
def local_file(tmp_path):
    path = tmp_path / "f.py"
    path.write_text("x = 1\n")
    return path


def test_run_on_host_builds_ssh_command_and_returns_result(ssh_run):
    ssh_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
    assert run_on_host("1.2.3.4", "ubuntu", KEY, "echo ok") == (0, "ok\n", "")
    argv = ssh_run.call_args[0][0]
    assert argv[0] == "ssh"
    assert "ubuntu@1.2.3.4" in argv
    assert argv[-1] == "echo ok"
    assert "BatchMode=yes" in argv


def test_run_on_host_custom_port(ssh_run):
    run_on_host("1.2.3.4", "ubuntu", KEY, "true", port=2222)
    argv = ssh_run.call_args[0][0]
    assert "-p" in argv and "2222" in argv


def test_run_on_host_timeout_returns_nonzero():
    with patch("aws_bootstrap.ssh.subprocess.run", side_effect=subprocess.TimeoutExpired("ssh", 5)):
        rc, _out, err = run_on_host("1.2.3.4", "ubuntu", KEY, "sleep 100", timeout=5)
    assert rc != 0
    assert "timed out" in err.lower()


def test_scp_to_host_builds_scp_command(ssh_run, local_file):
    assert scp_to_host("1.2.3.4", "ubuntu", KEY, local_file, "/tmp/f.py") is True
    argv = ssh_run.call_args[0][0]
    assert argv[0] == "scp"
    assert str(local_file) in argv
    assert "ubuntu@1.2.3.4:/tmp/f.py" in argv


def test_scp_to_host_failure_returns_false(ssh_run, local_file):
    ssh_run.return_value = MagicMock(returncode=1, stdout="", stderr="permission denied")
    assert scp_to_host("1.2.3.4", "ubuntu", KEY, local_file, "/tmp/f.py") is False


def test_scp_to_host_times_out_instead_of_hanging(tmp_path, local_file):
    """Regression: scp had no timeout, so one wedged transfer stalled the whole
    cluster fan-out indefinitely."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("scp", 5)):
        assert scp_to_host("1.2.3.4", "ubuntu", tmp_path / "k", local_file, "/tmp/f.txt") is False


def test_ssh_opts_are_non_interactive_and_bounded():
    """BatchMode keeps ssh/scp from blocking forever on a password prompt when
    the key does not match; ConnectTimeout bounds an unreachable host."""
    opts = _ssh_opts(Path("/k/id_ed25519.pub"))
    assert "BatchMode=yes" in opts
    assert f"ConnectTimeout={SSH_CONNECT_TIMEOUT}" in opts
    assert "/k/id_ed25519" in opts  # .pub stripped


def test_ssh_opts_connect_timeout_is_overridable():
    """ssh honors the FIRST -o occurrence, so a caller-specific timeout has to
    be threaded through here rather than appended after these options."""
    assert "ConnectTimeout=45" in _ssh_opts(Path("/k/id_ed25519"), connect_timeout=45)


@pytest.mark.parametrize(
    ("tool", "port", "expected"),
    [("ssh", 22, []), ("scp", 22, []), ("ssh", 2222, ["-p", "2222"]), ("scp", 2222, ["-P", "2222"])],
)
def test_port_opts(tool, port, expected):
    assert _port_opts(tool, port) == expected
