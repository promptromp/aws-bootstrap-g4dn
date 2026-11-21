"""Shared pytest fixtures for the aws_bootstrap CLI tests."""

from __future__ import annotations
from unittest.mock import patch

import pytest
from click.testing import CliRunner


@pytest.fixture(autouse=True)
def _stub_key_check(request):
    """Stub the CLI's pre-launch SSH key check.

    Most launch tests write only a ``.pub`` into tmp_path, so the real check
    (private key present, no passphrase or loaded in an agent) would stop them
    before the code under test. Tests of the check itself carry
    ``@pytest.mark.real_key_check``, or call ``ssh.check_key_usable`` directly.
    """
    if request.node.get_closest_marker("real_key_check"):
        yield None
        return
    with patch("aws_bootstrap.cli.check_key_usable") as stub:
        yield stub


@pytest.fixture
def runner():
    """A Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def cli_session():
    """Patch ``cli.boto3.Session`` and yield the mock.

    ``region_name`` is ``None`` so region resolution falls back to the
    ``us-west-2`` default unless a test passes explicit ``--region`` values.
    Per-region clients (``session.client("ec2"|"service-quotas", region_name=...)``)
    are plain MagicMocks; the helper functions that consume them are patched
    individually in each test.
    """
    with patch("aws_bootstrap.cli.boto3.Session") as mock_session:
        mock_session.return_value.region_name = None
        yield mock_session


@pytest.fixture
def quota_rows():
    """One spot + one on-demand G/VT quota row, as returned by get_family_quotas."""
    return [
        {
            "quota_code": "L-3819A6DF",
            "quota_name": "All G and VT Spot Instance Requests",
            "value": 0.0,
            "quota_type": "spot",
            "family": "gvt",
        },
        {
            "quota_code": "L-DB2E81BA",
            "quota_name": "Running On-Demand G and VT instances",
            "value": 8.0,
            "quota_type": "on-demand",
            "family": "gvt",
        },
    ]


@pytest.fixture
def instance_type_rows():
    """A single g4dn instance-type row, as returned by list_instance_types."""
    return [
        {
            "InstanceType": "g4dn.xlarge",
            "VCpuCount": 4,
            "MemoryMiB": 16384,
            "GpuSummary": "1x T4 (16384 MiB)",
        }
    ]


@pytest.fixture
def ami_rows():
    """A single AMI row, as returned by list_amis."""
    return [
        {
            "ImageId": "ami-abc123",
            "Name": "Deep Learning Base OSS Nvidia Driver GPU AMI",
            "CreationDate": "2025-06-01T00:00:00Z",
            "Architecture": "x86_64",
        }
    ]


@pytest.fixture
def history_rows():
    """A single quota-history row, as returned by get_quota_request_history."""
    return [
        {
            "request_id": "req-123",
            "status": "APPROVED",
            "quota_code": "L-3819A6DF",
            "quota_name": "All G and VT Spot Instance Requests",
            "desired_value": 8.0,
            "created": "2025-06-01T00:00:00Z",
        }
    ]
