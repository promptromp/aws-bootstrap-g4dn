"""ssh.check_key_usable: the pre-launch check that ssh can use the key without a prompt.

Keys are made with the real ``ssh-keygen`` in tmp_path. The agent is never the
user's: agent lookups are faked, and $SSH_AUTH_SOCK is cleared for every test.
"""

from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

import pytest

from aws_bootstrap import ssh
from aws_bootstrap.ec2 import CLIError


pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")


@pytest.fixture(autouse=True)
def _no_real_agent(monkeypatch):
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(ssh, "_agent_sockets", lambda: [])


def _keygen(tmp_path: Path, name: str, passphrase: str = "") -> Path:
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", passphrase, "-C", name, "-f", str(tmp_path / name)],
        check=True,
    )
    return tmp_path / f"{name}.pub"


@pytest.fixture
def no_keychain(tmp_path) -> Path:
    cfg = tmp_path / "ssh_config"
    cfg.write_text("Host *\n  ServerAliveInterval 60\n")
    return cfg


@pytest.fixture
def keychain(tmp_path) -> Path:
    cfg = tmp_path / "ssh_config_keychain"
    cfg.write_text("Host *\n  AddKeysToAgent yes\n  UseKeychain yes\n")
    return cfg


def test_key_without_passphrase_passes(tmp_path, no_keychain):
    ssh.check_key_usable(_keygen(tmp_path, "plain"), ssh_config=no_keychain)


def test_passphrase_key_outside_any_agent_fails_with_the_fix(tmp_path, no_keychain, monkeypatch):
    monkeypatch.setattr(ssh.platform, "system", lambda: "Darwin")
    with pytest.raises(CLIError) as exc:
        ssh.check_key_usable(_keygen(tmp_path, "locked", "s3cret"), ssh_config=no_keychain)
    msg = exc.value.format_message()
    assert "has a passphrase and isn't loaded in ssh-agent" in msg
    assert "BatchMode=yes" in msg
    assert f"ssh-add --apple-use-keychain {tmp_path / 'locked'}" in msg
    assert "--skip-key-check" in msg


def test_the_fix_is_plain_ssh_add_off_macos(tmp_path, no_keychain, monkeypatch):
    monkeypatch.setattr(ssh.platform, "system", lambda: "Linux")
    with pytest.raises(CLIError) as exc:
        ssh.check_key_usable(_keygen(tmp_path, "locked", "s3cret"), ssh_config=no_keychain)
    assert f"ssh-add {tmp_path / 'locked'}" in exc.value.format_message()
    assert "--apple-use-keychain" not in exc.value.format_message()


def test_passphrase_key_held_by_an_agent_passes(tmp_path, no_keychain, monkeypatch):
    pub = _keygen(tmp_path, "locked", "s3cret")
    fingerprint = ssh.key_fingerprint(pub)
    monkeypatch.setattr(ssh, "_agent_sockets", lambda: ["/tmp/fake-agent.sock"])
    real_run = ssh._run_key_tool

    def fake_run(cmd, env=None):
        if cmd[0] == "ssh-add":
            assert env["SSH_AUTH_SOCK"] == "/tmp/fake-agent.sock"
            return subprocess.CompletedProcess(cmd, 0, stdout=f"256 {fingerprint} locked (ED25519)\n", stderr="")
        return real_run(cmd, env)

    monkeypatch.setattr(ssh, "_run_key_tool", fake_run)
    ssh.check_key_usable(pub, ssh_config=no_keychain)


def test_agent_holding_other_keys_does_not_count(tmp_path, no_keychain, monkeypatch):
    pub = _keygen(tmp_path, "locked", "s3cret")
    monkeypatch.setattr(ssh, "_agent_sockets", lambda: ["/tmp/fake-agent.sock"])
    real_run = ssh._run_key_tool

    def fake_run(cmd, env=None):
        if cmd[0] == "ssh-add":
            return subprocess.CompletedProcess(cmd, 0, stdout="256 SHA256:someoneelse other (ED25519)\n", stderr="")
        return real_run(cmd, env)

    monkeypatch.setattr(ssh, "_run_key_tool", fake_run)
    with pytest.raises(CLIError):
        ssh.check_key_usable(pub, ssh_config=no_keychain)


def test_usekeychain_only_warns(tmp_path, keychain, capsys):
    """macOS may supply the passphrase from the Keychain without a prompt: undecidable, so warn."""
    ssh.check_key_usable(_keygen(tmp_path, "locked", "s3cret"), ssh_config=keychain)
    err = capsys.readouterr().err
    assert "Relying on the macOS Keychain" in err
    assert "ssh-add" in err


def test_missing_private_key_fails(tmp_path, no_keychain):
    pub = _keygen(tmp_path, "plain")
    (tmp_path / "plain").unlink()
    with pytest.raises(CLIError, match="SSH private key not found"):
        ssh.check_key_usable(pub, ssh_config=no_keychain)


def test_world_readable_private_key_fails_with_chmod_hint(tmp_path, no_keychain):
    pub = _keygen(tmp_path, "loose")
    (tmp_path / "loose").chmod(0o644)
    with pytest.raises(CLIError, match="chmod 600"):
        ssh.check_key_usable(pub, ssh_config=no_keychain)


def test_missing_ssh_tools_warn_and_continue(tmp_path, no_keychain, monkeypatch, capsys):
    """Can't decide without ssh-keygen: don't block a launch on that."""
    pub = _keygen(tmp_path, "locked", "s3cret")

    def no_tool(cmd, env=None):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(ssh, "_run_key_tool", no_tool)
    ssh.check_key_usable(pub, ssh_config=no_keychain)
    assert "could not check" in capsys.readouterr().err


def test_key_fingerprint_matches_ssh_keygen(tmp_path):
    pub = _keygen(tmp_path, "plain")
    out = subprocess.run(["ssh-keygen", "-l", "-E", "sha256", "-f", str(pub)], capture_output=True, text=True)
    assert ssh.key_fingerprint(pub) == out.stdout.split()[1]


def test_agent_sockets_include_an_identity_agent_from_ssh_config(monkeypatch):
    """1Password-style setups point ssh at an agent via IdentityAgent, not $SSH_AUTH_SOCK."""
    monkeypatch.undo()  # use the real _agent_sockets (the autouse fixture replaced it)
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/env.sock")

    def fake_run(cmd, env=None):
        assert cmd[:2] == ["ssh", "-G"]
        return subprocess.CompletedProcess(cmd, 0, stdout='user me\nidentityagent "~/agent.sock"\n', stderr="")

    monkeypatch.setattr(ssh, "_run_key_tool", fake_run)
    assert ssh._agent_sockets() == ["/run/env.sock", str(Path("~/agent.sock").expanduser())]


@pytest.mark.parametrize(
    "line,expected",
    [("  UseKeychain yes", True), ("UseKeychain=true", True), ("# UseKeychain yes", False), ("UseKeychain no", False)],
)
def test_uses_macos_keychain(tmp_path, line, expected):
    cfg = tmp_path / "cfg"
    cfg.write_text(f"Host *\n{line}\n")
    assert ssh._uses_macos_keychain(cfg) is expected


def test_uses_macos_keychain_missing_config(tmp_path):
    assert ssh._uses_macos_keychain(tmp_path / "nope") is False
