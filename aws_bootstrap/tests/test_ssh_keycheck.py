"""ssh.check_key_usable: the pre-launch check that ssh can use the key without a prompt.

Keys are made with the real ``ssh-keygen`` in tmp_path. The agent is never the
user's: agent lookups are faked, and $SSH_AUTH_SOCK is cleared for every test.
"""

from __future__ import annotations
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest

from aws_bootstrap import ssh
from aws_bootstrap.ec2 import CLIError


pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")

REAL_AGENT_SOCKETS = ssh._agent_sockets  # captured before the autouse fixture replaces it


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
def plain_key(tmp_path) -> Path:
    return _keygen(tmp_path, "plain")


@pytest.fixture
def locked_key(tmp_path) -> Path:
    return _keygen(tmp_path, "locked", "s3cret")


@pytest.fixture
def ssh_config(tmp_path):
    """Write an ssh config with the given lines; returns its path."""

    def write(*lines: str) -> Path:
        cfg = tmp_path / "ssh_config"
        cfg.write_text("Host *\n" + "".join(f"  {line}\n" for line in lines))
        return cfg

    return write


@pytest.fixture
def fake_agent(monkeypatch):
    """Make an agent at a fake socket list the given fingerprints (``ssh-add -l``)."""

    def install(*fingerprints: str) -> None:
        monkeypatch.setattr(ssh, "_agent_sockets", lambda: ["/tmp/fake-agent.sock"])
        real_run = ssh._run_key_tool

        def run(cmd, env=None):
            if cmd[0] != "ssh-add":
                return real_run(cmd, env)
            assert env["SSH_AUTH_SOCK"] == "/tmp/fake-agent.sock"
            out = "".join(f"256 {fp} key (ED25519)\n" for fp in fingerprints)
            return subprocess.CompletedProcess(cmd, 0 if fingerprints else 1, stdout=out, stderr="")

        monkeypatch.setattr(ssh, "_run_key_tool", run)

    return install


# ---------------------------------------------------------------------------
# check_key_usable
# ---------------------------------------------------------------------------


def test_key_without_passphrase_passes(plain_key, ssh_config):
    ssh.check_key_usable(plain_key, ssh_config=ssh_config())


@pytest.mark.parametrize(
    "system,fix,absent",
    [("Darwin", "ssh-add --apple-use-keychain", None), ("Linux", "ssh-add", "--apple-use-keychain")],
)
def test_passphrase_key_outside_any_agent_fails_with_the_fix(locked_key, ssh_config, monkeypatch, system, fix, absent):
    monkeypatch.setattr(ssh.platform, "system", lambda: system)
    with pytest.raises(CLIError) as exc:
        ssh.check_key_usable(locked_key, ssh_config=ssh_config())
    msg = exc.value.format_message()
    assert "has a passphrase and isn't loaded in ssh-agent" in msg
    assert "BatchMode=yes" in msg
    assert f"{fix} {locked_key.with_suffix('')}" in msg
    assert "--skip-key-check" in msg
    if absent:
        assert absent not in msg


PASSES = nullcontext()


def fails(match: str | None = None):
    return pytest.raises(CLIError, match=match)


@pytest.mark.parametrize(
    "holds_it,expectation",
    [(True, PASSES), (False, fails())],
    ids=["agent-holds-key", "agent-holds-other-keys"],
)
def test_passphrase_key_and_the_agent(locked_key, ssh_config, fake_agent, holds_it, expectation):
    fake_agent(ssh.key_fingerprint(locked_key) if holds_it else "SHA256:someoneelse")
    with expectation:
        ssh.check_key_usable(locked_key, ssh_config=ssh_config())


@pytest.mark.parametrize("system,expectation", [("Darwin", PASSES), ("Linux", fails("has a passphrase"))])
def test_usekeychain_only_warns_on_macos(locked_key, ssh_config, monkeypatch, capsys, system, expectation):
    """The Keychain may supply the passphrase on macOS (undecidable: warn); elsewhere nothing will."""
    monkeypatch.setattr(ssh.platform, "system", lambda: system)
    with expectation:
        ssh.check_key_usable(locked_key, ssh_config=ssh_config("IgnoreUnknown UseKeychain", "UseKeychain yes"))
    assert ("Relying on the macOS Keychain" in capsys.readouterr().err) is (system == "Darwin")


@pytest.mark.parametrize(
    "in_agent,expectation",
    [(False, fails("SSH private key not found")), (True, PASSES)],
    ids=["no-agent", "agent-holds-key"],
)
def test_missing_private_key(plain_key, ssh_config, fake_agent, in_agent, expectation):
    """Only the agent's copy can work without the file (e.g. 1Password): pass then, fail otherwise."""
    plain_key.with_suffix("").unlink()
    fake_agent(*([ssh.key_fingerprint(plain_key)] if in_agent else []))
    with expectation:
        ssh.check_key_usable(plain_key, ssh_config=ssh_config())


@pytest.mark.parametrize(
    "delete_private,expectation",
    [(False, PASSES), (True, fails("SSH private key not found"))],
    ids=["plain-key-passes", "missing-key-still-fails"],
)
def test_unreachable_agent_falls_through_to_the_key_file(
    plain_key, ssh_config, monkeypatch, delete_private, expectation
):
    """A stalled `ssh -G` / locked agent must not wave a missing or locked key through."""

    def stalled(fingerprint):
        raise subprocess.TimeoutExpired("ssh", 15)

    monkeypatch.setattr(ssh, "_agent_has_key", stalled)
    if delete_private:
        plain_key.with_suffix("").unlink()
    with expectation:
        ssh.check_key_usable(plain_key, ssh_config=ssh_config())


def test_world_readable_private_key_fails_with_chmod_hint(plain_key, ssh_config):
    plain_key.with_suffix("").chmod(0o644)
    with pytest.raises(CLIError, match="chmod 600"):
        ssh.check_key_usable(plain_key, ssh_config=ssh_config())


def test_not_an_openssh_public_key(tmp_path, ssh_config):
    bogus = tmp_path / "id.pub"
    bogus.write_text("-----BEGIN PUBLIC KEY-----\nnot*base64\n")
    with pytest.raises(CLIError, match="not an OpenSSH public key"):
        ssh.check_key_usable(bogus, ssh_config=ssh_config())


def test_missing_ssh_tools_warn_and_continue(locked_key, ssh_config, monkeypatch, capsys):
    """Can't decide without ssh-keygen: don't block a launch on that."""

    def no_tool(cmd, env=None):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(ssh, "_run_key_tool", no_tool)
    ssh.check_key_usable(locked_key, ssh_config=ssh_config())
    assert "could not check" in capsys.readouterr().err


def test_key_fingerprint_matches_ssh_keygen(plain_key):
    out = subprocess.run(["ssh-keygen", "-l", "-E", "sha256", "-f", str(plain_key)], capture_output=True, text=True)
    assert ssh.key_fingerprint(plain_key) == out.stdout.split()[1]


# ---------------------------------------------------------------------------
# which agent ssh asks (ssh_config(5): IdentityAgent overrides $SSH_AUTH_SOCK)
# ---------------------------------------------------------------------------


GROUP_CONTAINER_SOCK = "/Users/me/Library/Group Containers/2BUA8C4S2C.com.1password/t/agent.sock"


@pytest.mark.parametrize(
    "ssh_g_line,expected",
    [
        (None, ["/run/env.sock"]),
        (f"identityagent {GROUP_CONTAINER_SOCK}", [GROUP_CONTAINER_SOCK]),
        ("identityagent none", []),
        ("identityagent SSH_AUTH_SOCK", ["/run/env.sock"]),
        ("identityagent $MY_SOCK", ["/run/my.sock"]),
    ],
    ids=["no-identityagent", "path-with-spaces", "none", "literal-ssh-auth-sock", "env-var"],
)
def test_agent_sockets_follow_ssh_precedence(monkeypatch, ssh_g_line, expected):
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/env.sock")
    monkeypatch.setenv("MY_SOCK", "/run/my.sock")
    out = "user me\n" + (f"{ssh_g_line}\n" if ssh_g_line else "")
    monkeypatch.setattr(ssh, "_run_key_tool", lambda cmd, env=None: subprocess.CompletedProcess(cmd, 0, out, ""))
    assert REAL_AGENT_SOCKETS() == expected


@pytest.mark.parametrize(
    "line,expected",
    [
        ("UseKeychain yes", True),
        ("UseKeychain=true", True),
        ("# UseKeychain yes", False),
        ("UseKeychain no", False),
        (None, False),  # no config file at all
    ],
)
def test_uses_macos_keychain(tmp_path, ssh_config, line, expected):
    cfg = ssh_config(line) if line else tmp_path / "missing"
    assert ssh._uses_macos_keychain(cfg) is expected
