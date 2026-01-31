"""Tests for SSH config management (add/remove/find/list host stanzas)."""

from __future__ import annotations
import os
import stat
from pathlib import Path

from aws_bootstrap.ssh import (
    _next_alias,
    _read_ssh_config,
    add_ssh_host,
    find_ssh_alias,
    get_ssh_host_details,
    list_ssh_hosts,
    remove_ssh_host,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY_PATH = Path("/home/user/.ssh/id_ed25519.pub")


def _config_path(tmp_path: Path) -> Path:
    """Return a config path inside tmp_path (doesn't create the file)."""
    return tmp_path / ".ssh" / "config"


# ---------------------------------------------------------------------------
# Stanza creation
# ---------------------------------------------------------------------------


def test_add_creates_stanza(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    assert "# >>> aws-bootstrap [i-abc123] >>>" in content
    assert "# <<< aws-bootstrap [i-abc123] <<<" in content
    assert "Host aws-gpu1" in content
    assert "HostName 1.2.3.4" in content
    assert "User ubuntu" in content
    assert "IdentityFile /home/user/.ssh/id_ed25519" in content


def test_add_returns_alias(tmp_path):
    cfg = _config_path(tmp_path)
    alias = add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    assert alias == "aws-gpu1"


def test_add_uses_private_key_path(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", Path("/keys/mykey.pub"), config_path=cfg)
    content = cfg.read_text()
    assert "IdentityFile /keys/mykey" in content
    assert ".pub" not in content.split("IdentityFile")[1].split("\n")[0]


def test_add_includes_user(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ec2-user", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    assert "User ec2-user" in content


def test_add_includes_strict_host_checking(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    assert "StrictHostKeyChecking no" in content
    assert "UserKnownHostsFile /dev/null" in content


# ---------------------------------------------------------------------------
# Multiple instances
# ---------------------------------------------------------------------------


def test_second_host_gets_gpu2(tmp_path):
    cfg = _config_path(tmp_path)
    a1 = add_ssh_host("i-111", "1.1.1.1", "ubuntu", KEY_PATH, config_path=cfg)
    a2 = add_ssh_host("i-222", "2.2.2.2", "ubuntu", KEY_PATH, config_path=cfg)
    assert a1 == "aws-gpu1"
    assert a2 == "aws-gpu2"


def test_next_alias_empty():
    assert _next_alias("") == "aws-gpu1"


def test_next_alias_custom_prefix():
    assert _next_alias("", prefix="dev-box") == "dev-box1"


def test_next_alias_skips_user_hosts():
    """User-defined hosts with similar names should be ignored."""
    content = "Host aws-gpu99\n    HostName 1.2.3.4\n"
    # No marker blocks, so this should be ignored
    assert _next_alias(content) == "aws-gpu1"


# ---------------------------------------------------------------------------
# Preserving existing config
# ---------------------------------------------------------------------------


def test_preserves_existing_stanzas(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    existing = "Host myserver\n    HostName 10.0.0.1\n    User admin\n"
    cfg.write_text(existing)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    assert "Host myserver" in content
    assert "HostName 10.0.0.1" in content
    assert "Host aws-gpu1" in content


def test_preserves_trailing_newline(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("Host foo\n    HostName bar\n")
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    # Should not have triple+ blank lines
    assert "\n\n\n" not in content


def test_add_to_nonexistent_creates_dir_and_file(tmp_path):
    cfg = tmp_path / "brand_new" / ".ssh" / "config"
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    assert cfg.exists()
    assert cfg.parent.exists()


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def test_remove_returns_alias(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    removed = remove_ssh_host("i-abc123", config_path=cfg)
    assert removed == "aws-gpu1"


def test_remove_returns_none_when_missing(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    assert remove_ssh_host("i-missing", config_path=cfg) is None


def test_remove_preserves_other_stanzas(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-111", "1.1.1.1", "ubuntu", KEY_PATH, config_path=cfg)
    add_ssh_host("i-222", "2.2.2.2", "ubuntu", KEY_PATH, config_path=cfg)
    remove_ssh_host("i-111", config_path=cfg)
    content = cfg.read_text()
    assert "i-111" not in content
    assert "Host aws-gpu2" in content
    assert "HostName 2.2.2.2" in content


def test_remove_preserves_user_config(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("Host myserver\n    HostName 10.0.0.1\n")
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    remove_ssh_host("i-abc123", config_path=cfg)
    content = cfg.read_text()
    assert "Host myserver" in content
    assert "HostName 10.0.0.1" in content
    assert "aws-gpu" not in content


def test_remove_cleans_trailing_blanks(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    remove_ssh_host("i-abc123", config_path=cfg)
    content = cfg.read_text()
    assert "\n\n\n" not in content


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_add_idempotent_updates_ip(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    add_ssh_host("i-abc123", "5.6.7.8", "ubuntu", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    assert "HostName 5.6.7.8" in content
    assert "HostName 1.2.3.4" not in content


def test_add_idempotent_preserves_alias(tmp_path):
    cfg = _config_path(tmp_path)
    a1 = add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    a2 = add_ssh_host("i-abc123", "5.6.7.8", "ubuntu", KEY_PATH, config_path=cfg)
    assert a1 == a2 == "aws-gpu1"


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_find_alias_returns_alias(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    assert find_ssh_alias("i-abc123", config_path=cfg) == "aws-gpu1"


def test_find_alias_returns_none(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    assert find_ssh_alias("i-missing", config_path=cfg) is None


def test_list_hosts_returns_all(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-111", "1.1.1.1", "ubuntu", KEY_PATH, config_path=cfg)
    add_ssh_host("i-222", "2.2.2.2", "ubuntu", KEY_PATH, config_path=cfg)
    hosts = list_ssh_hosts(config_path=cfg)
    assert hosts == {"i-111": "aws-gpu1", "i-222": "aws-gpu2"}


def test_list_hosts_ignores_user_stanzas(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("Host myserver\n    HostName 10.0.0.1\n")
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    hosts = list_ssh_hosts(config_path=cfg)
    assert "myserver" not in hosts.values()
    assert hosts == {"i-abc123": "aws-gpu1"}


# ---------------------------------------------------------------------------
# Safety / edge cases
# ---------------------------------------------------------------------------


def test_file_permissions_0600(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    mode = stat.S_IMODE(os.stat(cfg).st_mode)
    assert mode == 0o600


def test_dir_permissions_0700(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    mode = stat.S_IMODE(os.stat(cfg.parent).st_mode)
    assert mode == 0o700


def test_handles_empty_file(tmp_path):
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    alias = add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    assert alias == "aws-gpu1"
    assert "Host aws-gpu1" in cfg.read_text()


def test_malformed_marker_left_alone(tmp_path):
    """Orphaned begin marker without end marker should not cause deletion."""
    cfg = _config_path(tmp_path)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    orphaned = "# >>> aws-bootstrap [i-orphan] >>>\nHost aws-gpu99\n    HostName 9.9.9.9\n"
    cfg.write_text(orphaned)
    removed = remove_ssh_host("i-orphan", config_path=cfg)
    assert removed is None
    content = cfg.read_text()
    assert "aws-gpu99" in content


def test_read_nonexistent_returns_empty(tmp_path):
    cfg = tmp_path / "does_not_exist"
    assert _read_ssh_config(cfg) == ""


def test_list_hosts_nonexistent_file(tmp_path):
    cfg = tmp_path / "no_such_file"
    assert list_ssh_hosts(config_path=cfg) == {}


def test_remove_nonexistent_file(tmp_path):
    cfg = tmp_path / "no_such_file"
    assert remove_ssh_host("i-abc123", config_path=cfg) is None


# ---------------------------------------------------------------------------
# Port in stanza / details
# ---------------------------------------------------------------------------


def test_stanza_includes_port_when_non_default(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg, port=2222)
    content = cfg.read_text()
    assert "Port 2222" in content


def test_stanza_omits_port_when_default(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    content = cfg.read_text()
    assert "Port" not in content


def test_get_ssh_host_details_parses_port(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg, port=2222)
    details = get_ssh_host_details("i-abc123", config_path=cfg)
    assert details is not None
    assert details.port == 2222


def test_get_ssh_host_details_default_port(tmp_path):
    cfg = _config_path(tmp_path)
    add_ssh_host("i-abc123", "1.2.3.4", "ubuntu", KEY_PATH, config_path=cfg)
    details = get_ssh_host_details("i-abc123", config_path=cfg)
    assert details is not None
    assert details.port == 22
