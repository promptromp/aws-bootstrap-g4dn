"""Tests for SSH config management (add/remove/find/list host stanzas)."""

from __future__ import annotations
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from aws_bootstrap.ssh import (
    _is_instance_id,
    _next_alias,
    _read_ssh_config,
    _write_ssh_config,
    add_ssh_host,
    find_drifted_ssh_hosts,
    find_missing_ssh_hosts,
    find_stale_ssh_hosts,
    get_ssh_host_details,
    list_ssh_hosts,
    remove_ssh_host,
    resolve_instance_id,
)


KEY_PATH = Path("/home/user/.ssh/id_ed25519.pub")
USER_STANZA = "Host myserver\n    HostName 10.0.0.1\n"


@pytest.fixture
def cfg(tmp_path) -> Path:
    """A config path inside tmp_path (the file is not created)."""
    return tmp_path / ".ssh" / "config"


@pytest.fixture
def write_cfg(cfg):
    """Create ``cfg`` with the given content (e.g. a user's own stanzas)."""

    def write(content: str = "") -> Path:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(content)
        return cfg

    return write


def _add(cfg, instance_id="i-abc123", ip="1.2.3.4", *, user="ubuntu", key=KEY_PATH, **kw):
    return add_ssh_host(instance_id, ip, user, key, config_path=cfg, **kw)


# ---------------------------------------------------------------------------
# Stanza creation
# ---------------------------------------------------------------------------


def test_add_creates_stanza(cfg):
    assert _add(cfg) == "aws-gpu1"
    content = cfg.read_text()
    assert "# >>> aws-bootstrap [i-abc123] >>>" in content
    assert "# <<< aws-bootstrap [i-abc123] <<<" in content
    assert "Host aws-gpu1" in content
    assert "HostName 1.2.3.4" in content
    assert "User ubuntu" in content
    assert "IdentityFile /home/user/.ssh/id_ed25519" in content
    assert "StrictHostKeyChecking no" in content
    assert "UserKnownHostsFile /dev/null" in content


def test_add_includes_user(cfg):
    _add(cfg, user="ec2-user")
    assert "User ec2-user" in cfg.read_text()


def test_add_uses_private_key_path(cfg):
    _add(cfg, key=Path("/keys/mykey.pub"))
    content = cfg.read_text()
    assert "IdentityFile /keys/mykey" in content
    assert ".pub" not in content.split("IdentityFile")[1].split("\n")[0]


def test_add_ssh_host_explicit_alias(cfg):
    assert _add(cfg, "i-123abcde", alias="aws-ml1-0") == "aws-ml1-0"
    assert "Host aws-ml1-0" in cfg.read_text()


def test_second_host_gets_gpu2(cfg):
    assert _add(cfg, "i-111", "1.1.1.1") == "aws-gpu1"
    assert _add(cfg, "i-222", "2.2.2.2") == "aws-gpu2"


@pytest.mark.parametrize(
    "content,prefix,expected",
    [
        ("", "aws-gpu", "aws-gpu1"),
        ("", "dev-box", "dev-box1"),
        # User-defined hosts with similar names (no marker blocks) are ignored.
        ("Host aws-gpu99\n    HostName 1.2.3.4\n", "aws-gpu", "aws-gpu1"),
    ],
    ids=["empty", "custom-prefix", "skips-user-hosts"],
)
def test_next_alias(content, prefix, expected):
    assert _next_alias(content, prefix=prefix) == expected


def test_add_idempotent_updates_ip_and_keeps_alias(cfg):
    a1 = _add(cfg, ip="1.2.3.4")
    a2 = _add(cfg, ip="5.6.7.8")
    assert a1 == a2 == "aws-gpu1"
    content = cfg.read_text()
    assert "HostName 5.6.7.8" in content
    assert "HostName 1.2.3.4" not in content


def test_stanza_includes_port_when_non_default(cfg):
    _add(cfg, port=2222)
    assert "Port 2222" in cfg.read_text()


def test_stanza_omits_port_when_default(cfg):
    _add(cfg)
    assert "Port" not in cfg.read_text()


# ---------------------------------------------------------------------------
# Preserving existing config / file handling
# ---------------------------------------------------------------------------


def test_preserves_existing_stanzas(write_cfg):
    cfg = write_cfg("Host myserver\n    HostName 10.0.0.1\n    User admin\n")
    _add(cfg)
    content = cfg.read_text()
    assert "Host myserver" in content
    assert "HostName 10.0.0.1" in content
    assert "Host aws-gpu1" in content


def test_add_to_nonexistent_creates_dir_and_file(tmp_path):
    cfg = tmp_path / "brand_new" / ".ssh" / "config"
    _add(cfg)
    assert cfg.exists()


def test_handles_empty_file(write_cfg):
    cfg = write_cfg("")
    assert _add(cfg) == "aws-gpu1"
    assert "Host aws-gpu1" in cfg.read_text()


def test_config_file_0600_and_dir_0700(cfg):
    _add(cfg)
    assert stat.S_IMODE(os.stat(cfg).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(cfg.parent).st_mode) == 0o700


@pytest.mark.parametrize(
    ("existing", "expected_prefix"),
    [
        ("", ""),
        ("Host other\n  HostName 9.9.9.9", "Host other\n  HostName 9.9.9.9\n\n"),
        ("Host other\n", "Host other\n\n"),
        ("Host other\n\n", "Host other\n\n"),
    ],
    ids=["empty", "no-trailing-newline", "one-newline", "already-blank-line"],
)
def test_add_ssh_host_separates_block_with_exactly_one_blank_line(cfg, write_cfg, existing, expected_prefix):
    if existing:
        write_cfg(existing)
    _add(cfg, "i-111aaaa1", "1.1.1.1")
    content = cfg.read_text()
    assert content.startswith(expected_prefix)
    assert "\n\n\n" not in content


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def test_remove_returns_alias_and_cleans_trailing_blanks(cfg):
    _add(cfg)
    assert remove_ssh_host("i-abc123", config_path=cfg) == "aws-gpu1"
    assert "\n\n\n" not in cfg.read_text()


def test_remove_returns_none_when_missing(write_cfg):
    assert remove_ssh_host("i-missing", config_path=write_cfg("")) is None


def test_remove_preserves_other_stanzas(cfg):
    _add(cfg, "i-111", "1.1.1.1")
    _add(cfg, "i-222", "2.2.2.2")
    remove_ssh_host("i-111", config_path=cfg)
    content = cfg.read_text()
    assert "i-111" not in content
    assert "Host aws-gpu2" in content
    assert "HostName 2.2.2.2" in content


def test_remove_preserves_user_config(write_cfg):
    cfg = write_cfg(USER_STANZA)
    _add(cfg)
    remove_ssh_host("i-abc123", config_path=cfg)
    content = cfg.read_text()
    assert "Host myserver" in content
    assert "HostName 10.0.0.1" in content
    assert "aws-gpu" not in content


def test_malformed_marker_left_alone(write_cfg):
    """Orphaned begin marker without end marker should not cause deletion."""
    cfg = write_cfg("# >>> aws-bootstrap [i-orphan] >>>\nHost aws-gpu99\n    HostName 9.9.9.9\n")
    assert remove_ssh_host("i-orphan", config_path=cfg) is None
    assert "aws-gpu99" in cfg.read_text()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_list_hosts_returns_all(cfg):
    _add(cfg, "i-111", "1.1.1.1")
    _add(cfg, "i-222", "2.2.2.2")
    assert list_ssh_hosts(config_path=cfg) == {"i-111": "aws-gpu1", "i-222": "aws-gpu2"}


def test_list_hosts_ignores_user_stanzas(write_cfg):
    cfg = write_cfg(USER_STANZA)
    _add(cfg)
    assert list_ssh_hosts(config_path=cfg) == {"i-abc123": "aws-gpu1"}


@pytest.mark.parametrize("port", [2222, 22], ids=["non-default", "default"])
def test_get_ssh_host_details_port(cfg, port):
    _add(cfg, port=port)
    details = get_ssh_host_details("i-abc123", config_path=cfg)
    assert details is not None
    assert details.port == port


@pytest.mark.parametrize(
    "call,expected",
    [
        pytest.param(_read_ssh_config, "", id="read"),
        pytest.param(lambda p: list_ssh_hosts(config_path=p), {}, id="list"),
        pytest.param(lambda p: remove_ssh_host("i-abc123", config_path=p), None, id="remove"),
        pytest.param(lambda p: resolve_instance_id("aws-gpu1", config_path=p), None, id="resolve"),
    ],
)
def test_missing_config_file_reads_as_empty(tmp_path, call, expected):
    assert call(tmp_path / "no_such_file") == expected


# ---------------------------------------------------------------------------
# Instance ID detection / resolve_instance_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("i-abcdef01", True),
        ("i-0123456789abcdef0", True),
        ("aws-gpu1", False),
        ("", False),
        ("i-", False),
        ("i-ABCDEF01", False),
        ("i-abc", False),
    ],
    ids=["short", "long", "alias", "empty", "prefix-only", "uppercase", "too-short"],
)
def test_is_instance_id(value, expected):
    assert _is_instance_id(value) is expected


def test_resolve_passthrough_instance_id(write_cfg):
    """Instance IDs are returned as-is without consulting SSH config."""
    assert resolve_instance_id("i-0123456789abcdef0", config_path=write_cfg("")) == "i-0123456789abcdef0"


def test_resolve_alias_to_instance_id(cfg):
    _add(cfg, "i-111aaa11", "1.1.1.1")
    _add(cfg, "i-222bbb22", "2.2.2.2")
    assert resolve_instance_id("aws-gpu1", config_path=cfg) == "i-111aaa11"
    assert resolve_instance_id("aws-gpu2", config_path=cfg) == "i-222bbb22"


def test_resolve_unknown_alias_returns_none(write_cfg):
    assert resolve_instance_id("aws-gpu99", config_path=write_cfg("")) is None


# ---------------------------------------------------------------------------
# Cleanup helpers: stale / missing / drifted (for `cleanup` and `cleanup --sync`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "live,expected",
    [
        ({"i-111aaaa1"}, [("i-222bbbb2", "aws-gpu2")]),
        ({"i-111aaaa1", "i-222bbbb2"}, []),
        (set(), [("i-111aaaa1", "aws-gpu1"), ("i-222bbbb2", "aws-gpu2")]),
    ],
    ids=["some-stale", "none-stale", "all-stale"],
)
def test_find_stale_ssh_hosts(cfg, live, expected):
    _add(cfg, "i-111aaaa1", "1.1.1.1")
    _add(cfg, "i-222bbbb2", "2.2.2.2")
    assert sorted(find_stale_ssh_hosts(live, config_path=cfg)) == expected


def test_find_stale_ssh_hosts_empty_config(write_cfg):
    assert find_stale_ssh_hosts(set(), config_path=write_cfg("")) == []


def test_find_missing_ssh_hosts(cfg):
    _add(cfg, "i-1111aaaa", "1.1.1.1")
    live = [
        {"InstanceId": "i-1111aaaa", "PublicIp": "1.1.1.1"},
        {"InstanceId": "i-2222bbbb", "PublicIp": "2.2.2.2"},
        {"InstanceId": "i-3333cccc", "PublicIp": ""},  # no public IP -> nothing to point an alias at
    ]
    assert [m["InstanceId"] for m in find_missing_ssh_hosts(live, config_path=cfg)] == ["i-2222bbbb"]


@pytest.mark.parametrize(
    "live_ip,expected",
    [("9.9.9.9", [("i-1", "aws-gpu1", "9.9.9.9")]), ("1.1.1.1", [])],
    ids=["ip-changed", "ip-matches"],
)
def test_find_drifted_ssh_hosts(cfg, live_ip, expected):
    _add(cfg, "i-1", "1.1.1.1")
    assert find_drifted_ssh_hosts([{"InstanceId": "i-1", "PublicIp": live_ip}], config_path=cfg) == expected


# ---------------------------------------------------------------------------
# Atomic write: the error path must surface the real cause
# ---------------------------------------------------------------------------


def test_write_ssh_config_failure_surfaces_the_real_cause_and_cleans_up(tmp_path):
    """Regression: the cleanup handler called os.get_inheritable() on an
    already-closed fd, replacing the real error with 'Bad file descriptor'."""
    with (
        patch("os.replace", side_effect=PermissionError("read-only fs")),
        pytest.raises(PermissionError, match="read-only fs"),
    ):
        _write_ssh_config(tmp_path / "config", "Host x\n")
    assert list(tmp_path.glob(".ssh_config_tmp_*")) == []  # temp file removed


def test_write_ssh_config_writes_content_and_mode(tmp_path):
    cfg = tmp_path / "config"
    _write_ssh_config(cfg, "Host x\n")
    assert cfg.read_text() == "Host x\n"
    assert cfg.stat().st_mode & 0o777 == 0o600
