"""Shared pytest fixtures for the aws_bootstrap CLI tests."""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aws_bootstrap.ec2 import RegionContext, RegionLaunch


@pytest.fixture(autouse=True)
def key_check_stub(request):
    """Stub the CLI's pre-launch SSH key check.

    Most launch tests write only a ``.pub`` into tmp_path, so the real check
    (private key present, no passphrase or loaded in an agent) would stop them
    before the code under test. Tests of the check itself carry
    ``@pytest.mark.real_key_check``, or call ``ssh.check_key_usable`` directly.
    """
    if request.node.get_closest_marker("real_key_check"):
        # The real check, but never against the user's ssh-agent: no agent unless a test fakes one.
        with patch("aws_bootstrap.ssh._agent_sockets", return_value=[]):
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
def key_path(tmp_path):
    """A throwaway public key in tmp_path (no private half: the key check is stubbed)."""
    pub = tmp_path / "id_ed25519.pub"
    pub.write_text("ssh-ed25519 AAAA test@host")
    return pub


@pytest.fixture
def region_launch():
    """Factory for a ``RegionLaunch`` as ``launch_with_retry`` returns it."""

    def make(instance=None, *, region="us-west-2", pricing="spot", **ctx_fields):
        ctx = RegionContext(
            region=region,
            ec2_client=MagicMock(),
            ami={"ImageId": "ami-123", "Name": "TestAMI"},
            sg_id="sg-123",
            key_name="aws-bootstrap-key",
            **ctx_fields,
        )
        return RegionLaunch(
            region=region, context=ctx, instance=instance or {"InstanceId": "i-test123"}, pricing=pricing
        )

    return make


@pytest.fixture
def launch_aws(cli_session):
    """Patch the per-region setup ``launch`` does before it launches (all a ``--dry-run`` needs).

    ``instance_type_architecture`` is deliberately left real: against the mocked
    client it falls back to ``x86_64``.
    """
    with (
        patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-123", "Name": "TestAMI"}) as ami,
        patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key") as import_key,
        patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123") as sg,
    ):
        yield SimpleNamespace(session=cli_session, ami=ami, import_key=import_key, sg=sg)


@pytest.fixture
def launched(launch_aws, region_launch):
    """Patch a real (non-dry-run) launch end to end.

    ``i-test123`` comes up in us-west-2a at 1.2.3.4, SSH answers, setup succeeds and
    the alias is ``aws-gpu1``. Override a mock's return value to steer a test down an
    incomplete path (``ready`` without ``PublicIpAddress``, ``ssh_ready`` False, ...).
    ``query_cuda_version`` is patched because, real, it ssh'd to 1.2.3.4 and waited
    out the 10s ConnectTimeout.
    """
    instance = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2a"}}
    with (
        patch("aws_bootstrap.cli.launch_with_retry", return_value=region_launch()) as launch,
        patch("aws_bootstrap.cli.wait_instance_ready", return_value=instance) as ready,
        patch("aws_bootstrap.cli.wait_for_ssh", return_value=True) as ssh_ready,
        patch("aws_bootstrap.cli.run_remote_setup", return_value=True) as setup,
        patch("aws_bootstrap.cli.query_cuda_version", return_value=None) as cuda,
        patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1") as add_ssh,
    ):
        yield SimpleNamespace(
            **vars(launch_aws), launch=launch, ready=ready, ssh_ready=ssh_ready, setup=setup, cuda=cuda, add_ssh=add_ssh
        )


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
