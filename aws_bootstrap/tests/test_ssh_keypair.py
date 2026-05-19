"""Tests for key-pair handling, SSH-failure classification, and key generation."""

from __future__ import annotations
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from aws_bootstrap.ec2 import CLIError
from aws_bootstrap.ssh import (
    _classify_ssh_failure,
    _derived_key_name,
    _pubkey_blob,
    generate_ssh_keypair,
    import_key_pair,
    wait_for_ssh,
)


LOCAL_PUB = "ssh-ed25519 AAAALOCALKEYBLOB adamhadani@Mac\n"
OTHER_PUB_BLOB = "AAAADIFFERENTKEYBLOB"


def _pub(tmp_path: Path) -> Path:
    p = tmp_path / "id_ed25519.pub"
    p.write_text(LOCAL_PUB)
    return p


def _ec2(describe_side_effect):
    c = MagicMock()
    c.exceptions.ClientError = botocore.exceptions.ClientError
    c.describe_key_pairs.side_effect = describe_side_effect
    return c


def _not_found():
    return botocore.exceptions.ClientError(
        {"Error": {"Code": "InvalidKeyPair.NotFound", "Message": "not found"}}, "DescribeKeyPairs"
    )


# --- pure helpers ------------------------------------------------------------


def test_pubkey_blob_strips_comment():
    assert _pubkey_blob("ssh-ed25519 AAAABLOB user@host") == "AAAABLOB"
    assert _pubkey_blob("AAAABLOB") == "AAAABLOB"


def test_derived_key_name_deterministic_and_collision_free():
    a = _derived_key_name("aws-bootstrap-key", "AAAABLOB")
    assert a == _derived_key_name("aws-bootstrap-key", "AAAABLOB")
    assert a != _derived_key_name("aws-bootstrap-key", "OTHERBLOB")
    assert a.startswith("aws-bootstrap-key-") and len(a.split("-")[-1]) == 8


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("Permission denied (publickey).", "auth"),
        ("kex_exchange_identification: Connection closed by remote host", "transient"),
        ("ssh: connect to host x port 22: Connection refused", "transient"),
        ("Too many authentication failures", "auth"),
        ("@@@ REMOTE HOST IDENTIFICATION HAS CHANGED! @@@", "auth"),
        ("", "transient"),
    ],
)
def test_classify_ssh_failure(stderr, expected):
    assert _classify_ssh_failure(stderr) == expected


# --- import_key_pair ---------------------------------------------------------


def test_import_key_pair_not_exists_imports_under_requested_name(tmp_path):
    ec2 = _ec2(_not_found())
    assert import_key_pair(ec2, "aws-bootstrap-key", _pub(tmp_path)) == "aws-bootstrap-key"
    ec2.import_key_pair.assert_called_once()
    assert ec2.import_key_pair.call_args[1]["KeyName"] == "aws-bootstrap-key"


def test_import_key_pair_matching_reuses_name(tmp_path):
    ec2 = _ec2([{"KeyPairs": [{"PublicKey": LOCAL_PUB}]}])
    assert import_key_pair(ec2, "aws-bootstrap-key", _pub(tmp_path)) == "aws-bootstrap-key"
    ec2.import_key_pair.assert_not_called()


def test_import_key_pair_no_pubkey_on_record_reuses(tmp_path):
    # Older imported pair: AWS returns no PublicKey -> reuse (assume it's ours).
    ec2 = _ec2([{"KeyPairs": [{}]}])
    assert import_key_pair(ec2, "aws-bootstrap-key", _pub(tmp_path)) == "aws-bootstrap-key"
    ec2.import_key_pair.assert_not_called()


def test_import_key_pair_mismatch_uses_derived_name_and_never_deletes(tmp_path):
    pub = _pub(tmp_path)
    local_blob = _pubkey_blob(LOCAL_PUB)
    derived = _derived_key_name("aws-bootstrap-key", local_blob)
    # 1st describe: name exists with a DIFFERENT key. 2nd describe (derived): not found.
    ec2 = _ec2([{"KeyPairs": [{"PublicKey": f"ssh-ed25519 {OTHER_PUB_BLOB} other"}]}, _not_found()])
    result = import_key_pair(ec2, "aws-bootstrap-key", pub)
    assert result == derived
    ec2.import_key_pair.assert_called_once()
    assert ec2.import_key_pair.call_args[1]["KeyName"] == derived
    # The pre-existing mismatched pair must never be modified/deleted.
    ec2.delete_key_pair.assert_not_called()


def test_import_key_pair_mismatch_reuses_existing_derived_when_matching(tmp_path):
    pub = _pub(tmp_path)
    ec2 = _ec2(
        [
            {"KeyPairs": [{"PublicKey": f"ssh-ed25519 {OTHER_PUB_BLOB} other"}]},
            {"KeyPairs": [{"PublicKey": LOCAL_PUB}]},  # derived already exists & matches us
        ]
    )
    result = import_key_pair(ec2, "aws-bootstrap-key", pub)
    assert result == _derived_key_name("aws-bootstrap-key", _pubkey_blob(LOCAL_PUB))
    ec2.import_key_pair.assert_not_called()


def test_import_key_pair_derived_collision_with_third_key_errors(tmp_path):
    pub = _pub(tmp_path)
    ec2 = _ec2(
        [
            {"KeyPairs": [{"PublicKey": f"ssh-ed25519 {OTHER_PUB_BLOB} other"}]},
            {"KeyPairs": [{"PublicKey": "ssh-ed25519 AAAATHIRDKEY third"}]},
        ]
    )
    with pytest.raises(CLIError, match="already exists in this region with a different key"):
        import_key_pair(ec2, "aws-bootstrap-key", pub)


# --- wait_for_ssh ------------------------------------------------------------


@patch("aws_bootstrap.ssh.socket.create_connection")
@patch("aws_bootstrap.ssh.subprocess.run")
def test_wait_for_ssh_auth_failure_fails_fast(mock_run, mock_sock):
    mock_sock.return_value = MagicMock()  # port open
    mock_run.return_value = MagicMock(returncode=255, stderr="Permission denied (publickey).")
    ok = wait_for_ssh("1.2.3.4", "ubuntu", Path("/tmp/k.pub"), retries=30, delay=0)
    assert ok is False
    # Fatal auth error must NOT consume the full retry budget.
    assert mock_run.call_count == 1


@patch("aws_bootstrap.ssh.time.sleep", lambda _s: None)
@patch("aws_bootstrap.ssh.socket.create_connection")
@patch("aws_bootstrap.ssh.subprocess.run")
def test_wait_for_ssh_transient_then_success(mock_run, mock_sock):
    mock_sock.return_value = MagicMock()
    mock_run.side_effect = [
        MagicMock(returncode=255, stderr="kex_exchange_identification: Connection closed"),
        MagicMock(returncode=0, stderr=""),
    ]
    assert wait_for_ssh("1.2.3.4", "ubuntu", Path("/tmp/k.pub"), retries=5, delay=0) is True
    assert mock_run.call_count == 2


# --- generate_ssh_keypair ----------------------------------------------------


def test_generate_ssh_keypair_invokes_ssh_keygen(tmp_path):
    pub = tmp_path / "sub" / "id_ed25519.pub"
    with patch("aws_bootstrap.ssh.subprocess.run") as mock_run:
        generate_ssh_keypair(pub)
    args = mock_run.call_args[0][0]
    assert args[0] == "ssh-keygen" and "ed25519" in args and str(tmp_path / "sub" / "id_ed25519") in args
    assert (tmp_path / "sub").is_dir()  # parent created


def test_generate_ssh_keypair_propagates_failure(tmp_path):
    with (
        patch(
            "aws_bootstrap.ssh.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ssh-keygen"),
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        generate_ssh_keypair(tmp_path / "id_ed25519.pub")
