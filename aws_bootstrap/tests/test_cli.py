"""Tests for CLI entry point and help output."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest
import yaml
from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.ec2 import CLIError, RegionContext, RegionLaunch
from aws_bootstrap.gpu import GpuInfo
from aws_bootstrap.ssh import SSHHostDetails, add_ssh_host


def _region_launch(instance, *, region="us-west-2", pricing="spot", ami=None):
    """Build a RegionLaunch as launch_with_retry would return it."""
    ami = ami or {"ImageId": "ami-123", "Name": "TestAMI"}
    ctx = RegionContext(region=region, ec2_client=MagicMock(), ami=ami, sg_id="sg-123", key_name="aws-bootstrap-key")
    return RegionLaunch(region=region, context=ctx, instance=instance, pricing=pricing)


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Bootstrap AWS EC2 GPU instances" in result.output
    assert "launch" in result.output
    assert "status" in result.output
    assert "terminate" in result.output
    assert "list" in result.output


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_launch_help():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--help"])
    assert result.exit_code == 0
    assert "--instance-type" in result.output
    assert "--spot" in result.output
    assert "--dry-run" in result.output
    assert "--key-path" in result.output


def test_launch_missing_key_ungeneratable_path_errors_clearly():
    # Missing key now triggers auto-generation; an ungeneratable path
    # (cannot create /nonexistent) must fail with a clear message.
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", "/nonexistent/key.pub"])
    assert result.exit_code != 0
    assert "could not be generated" in result.output


@patch("aws_bootstrap.cli.generate_ssh_keypair")
def test_launch_missing_key_dry_run_does_not_generate(mock_gen, tmp_path):
    # --dry-run must NOT touch the filesystem (and the notice would be
    # invisible under --output json) — report intent and stop instead.
    key = tmp_path / "id_ed25519.pub"
    result = CliRunner().invoke(main, ["launch", "--key-path", str(key), "--dry-run", "--region", "us-west-2"])
    assert result.exit_code != 0
    mock_gen.assert_not_called()
    assert not key.exists()
    assert "re-run without --dry-run" in result.output


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.generate_ssh_keypair")
def test_launch_missing_key_autogenerates_on_real_launch(
    mock_gen, mock_session, mock_launch, mock_wait, mock_ssh, mock_add_ssh, tmp_path
):
    mock_session.return_value.region_name = None
    key = tmp_path / "id_ed25519.pub"

    def _fake_gen(pub_path):
        Path(pub_path).write_text("ssh-ed25519 AAAAGEN generated\n")

    mock_gen.side_effect = _fake_gen
    mock_launch.return_value = _region_launch({"InstanceId": "i-xyz"}, region="us-west-2")
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2a"}}
    result = CliRunner().invoke(main, ["launch", "--key-path", str(key), "--no-setup", "--region", "us-west-2"])
    assert result.exit_code == 0
    mock_gen.assert_called_once()
    assert "generating a new ed25519 key pair" in result.output


def test_status_help():
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])
    assert result.exit_code == 0
    assert "--region" in result.output
    assert "--profile" in result.output


def test_terminate_help():
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--help"])
    assert result.exit_code == 0
    assert "--region" in result.output
    assert "--yes" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_no_instances(mock_find, mock_session):
    mock_find.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "No active" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_shows_instances(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.1578
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "i-abc123" in result.output
    assert "1.2.3.4" in result.output
    assert "spot ($0.1578/hr)" in result.output
    assert "Uptime" in result.output
    assert "Est. cost" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_terminate_hint_pins_resolved_region(
    mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details
):
    """'To terminate' hint must carry --region for the region status used."""
    mock_session.return_value.region_name = None
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-1a",
        }
    ]
    mock_spot_price.return_value = 0.1578
    result = CliRunner().invoke(main, ["status", "--region", "us-west-1"])
    assert result.exit_code == 0
    assert "aws-bootstrap terminate i-abc123 --region us-west-1" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_family_quotas")
def test_quota_show_request_hint_pins_resolved_region(mock_quotas, mock_session):
    """quota show's 'To request an increase' hint must carry --region so it
    targets the queried region, not the default (reported regression)."""
    mock_session.return_value.region_name = None
    mock_quotas.return_value = [
        {
            "quota_code": "L-3819A6DF",
            "quota_name": "All G and VT Spot",
            "value": 0.0,
            "quota_type": "spot",
            "family": "gvt",
        },
        {
            "quota_code": "L-DB2E81BA",
            "quota_name": "Running On-Demand G and VT",
            "value": 0.0,
            "quota_type": "on-demand",
            "family": "gvt",
        },
    ]
    result = CliRunner().invoke(main, ["quota", "show", "--family", "gvt", "--region", "us-west-1"])
    assert result.exit_code == 0
    # current spot quota is 0 -> suggested desired-value is max(8, 0+4) = 8,
    # and the command is pinned to the queried region.
    assert "aws-bootstrap quota request --family gvt --type spot --desired-value 8 --region us-west-1" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_family_quotas")
def test_quota_show_suggests_value_above_current(mock_quotas, mock_session):
    """Suggested --desired-value must exceed the current quota (AWS rejects <=)."""
    mock_session.return_value.region_name = None
    mock_quotas.return_value = [
        {
            "quota_code": "L-3819A6DF",
            "quota_name": "All G and VT Spot",
            "value": 32.0,
            "quota_type": "spot",
            "family": "gvt",
        },
        {
            "quota_code": "L-DB2E81BA",
            "quota_name": "Running On-Demand G and VT",
            "value": 8.0,
            "quota_type": "on-demand",
            "family": "gvt",
        },
    ]
    result = CliRunner().invoke(main, ["quota", "show", "--family", "gvt", "--region", "us-east-1"])
    assert result.exit_code == 0
    # current spot = 32 -> suggested = max(8, 32+4) = 36 (> current).
    assert "--type spot --desired-value 36 --region us-east-1" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_on_demand_no_cost(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details):
    mock_find.return_value = [
        {
            "InstanceId": "i-ondemand",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "5.6.7.8",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "on-demand",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "on-demand" in result.output
    assert "Uptime" not in result.output
    assert "Est. cost" not in result.output
    mock_spot_price.assert_not_called()


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_terminate_no_instances(mock_find, mock_session):
    mock_find.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["terminate"])
    assert result.exit_code == 0
    assert "No active" in result.output


@patch("aws_bootstrap.cli.remove_ssh_host", return_value=None)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
def test_terminate_with_confirm(mock_terminate, mock_find, mock_session, mock_remove_ssh):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes"])
    assert result.exit_code == 0
    assert "Terminated 1" in result.output
    mock_terminate.assert_called_once()
    assert mock_terminate.call_args[0][1] == ["i-abc123"]


@patch("aws_bootstrap.cli.remove_ssh_host", return_value=None)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
@patch("aws_bootstrap.cli.resolve_instance_id", return_value="i-abc123")
def test_terminate_by_alias(mock_resolve, mock_terminate, mock_session, mock_remove_ssh):
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes", "aws-gpu1"])
    assert result.exit_code == 0
    assert "Resolved alias 'aws-gpu1' -> i-abc123" in result.output
    assert "Terminated 1" in result.output
    mock_resolve.assert_called_once_with("aws-gpu1")
    mock_terminate.assert_called_once()
    assert mock_terminate.call_args[0][1] == ["i-abc123"]


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.resolve_instance_id", return_value=None)
def test_terminate_unknown_alias_errors(mock_resolve, mock_session):
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes", "aws-gpu99"])
    assert result.exit_code != 0
    assert "Could not resolve 'aws-gpu99'" in result.output


@patch("aws_bootstrap.cli.remove_ssh_host", return_value=None)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
@patch("aws_bootstrap.cli.resolve_instance_id", return_value="i-abc123")
def test_terminate_by_instance_id_passthrough(mock_resolve, mock_terminate, mock_session, mock_remove_ssh):
    """Instance IDs are passed through without resolution message."""
    mock_resolve.return_value = "i-abc123"
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes", "i-abc123"])
    assert result.exit_code == 0
    assert "Resolved alias" not in result.output
    assert "Terminated 1" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_terminate_cancelled(mock_find, mock_session):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["terminate"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output


# ---------------------------------------------------------------------------
# list subcommand
# ---------------------------------------------------------------------------


def test_list_help():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "--help"])
    assert result.exit_code == 0
    assert "instance-types" in result.output
    assert "amis" in result.output


def test_list_instance_types_help():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "instance-types", "--help"])
    assert result.exit_code == 0
    assert "--prefix" in result.output
    assert "--region" in result.output


def test_list_amis_help():
    runner = CliRunner()
    result = runner.invoke(main, ["list", "amis", "--help"])
    assert result.exit_code == 0
    assert "--filter" in result.output
    assert "--region" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.list_instance_types")
def test_list_instance_types_output(mock_list, mock_session):
    mock_list.return_value = [
        {
            "InstanceType": "g4dn.xlarge",
            "VCpuCount": 4,
            "MemoryMiB": 16384,
            "GpuSummary": "1x T4 (16384 MiB)",
        },
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["list", "instance-types"])
    assert result.exit_code == 0
    assert "g4dn.xlarge" in result.output
    assert "16384" in result.output
    assert "T4" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.list_instance_types")
def test_list_instance_types_empty(mock_list, mock_session):
    mock_list.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["list", "instance-types", "--prefix", "zzz"])
    assert result.exit_code == 0
    assert "No instance types found" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.list_amis")
def test_list_amis_output(mock_list, mock_session):
    mock_list.return_value = [
        {
            "ImageId": "ami-abc123",
            "Name": "Deep Learning AMI v42",
            "CreationDate": "2025-06-01T00:00:00Z",
            "Architecture": "x86_64",
        },
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["list", "amis"])
    assert result.exit_code == 0
    assert "ami-abc123" in result.output
    assert "Deep Learning AMI v42" in result.output
    assert "2025-06-01" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.list_amis")
def test_list_amis_empty(mock_list, mock_session):
    mock_list.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["list", "amis", "--filter", "nonexistent*"])
    assert result.exit_code == 0
    assert "No AMIs found" in result.output


# ---------------------------------------------------------------------------
# SSH config integration tests
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_output_shows_ssh_alias(
    mock_session, mock_ami, mock_import, mock_sg, mock_launch, mock_wait, mock_ssh, mock_setup, mock_add_ssh, tmp_path
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = _region_launch({"InstanceId": "i-test123"})
    mock_wait.return_value = {
        "PublicIpAddress": "1.2.3.4",
        "Placement": {"AvailabilityZone": "us-west-2a"},
    }

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--no-setup"])
    assert result.exit_code == 0
    assert "ssh aws-gpu1" in result.output
    assert "SSH alias: aws-gpu1" in result.output
    mock_add_ssh.assert_called_once()


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.add_ssh_host")
def test_launch_dry_run_no_ssh_config(mock_add_ssh, mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run"])
    assert result.exit_code == 0
    mock_add_ssh.assert_not_called()


@patch("aws_bootstrap.cli.remove_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
def test_terminate_removes_ssh_config(mock_terminate, mock_find, mock_session, mock_remove_ssh):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes"])
    assert result.exit_code == 0
    assert "Removed SSH config alias: aws-gpu1" in result.output
    mock_remove_ssh.assert_called_once_with("i-abc123")


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts")
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_shows_alias(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.15
    mock_ssh_hosts.return_value = {"i-abc123": "aws-gpu1"}
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "aws-gpu1" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_no_alias_graceful(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details):
    mock_find.return_value = [
        {
            "InstanceId": "i-old999",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "9.8.7.6",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.15
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "i-old999" in result.output


# ---------------------------------------------------------------------------
# multi-region status tests
# ---------------------------------------------------------------------------


def _region_instance(instance_id="i-east", region="us-east-1"):
    return {
        "InstanceId": instance_id,
        "Name": "aws-bootstrap-g4dn.xlarge",
        "State": "running",
        "InstanceType": "g4dn.xlarge",
        "PublicIp": "1.2.3.4",
        "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        "Lifecycle": "spot",
        "AvailabilityZone": f"{region}a",
        "Region": region,
    }


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["eu-west-1", "us-east-1", "us-west-2"])
def test_status_no_region_queries_all_enabled(
    mock_regions, mock_find, mock_session, mock_spot, mock_ssh_hosts, mock_details, mock_ebs
):
    """Naked status discovers all enabled regions and queries them."""
    mock_find.return_value = ([_region_instance(region="us-east-1")], [])
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    mock_regions.assert_called_once()
    assert mock_find.call_args[0][2] == ["eu-west-1", "us-east-1", "us-west-2"]
    assert "Querying 3 enabled region(s)" in result.output
    assert "us-east-1" in result.output


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.list_enabled_regions")
def test_status_selected_regions_filter(
    mock_regions, mock_find, mock_session, mock_spot, mock_ssh_hosts, mock_details, mock_ebs
):
    """Explicit --region values skip discovery and restrict the query."""
    mock_find.return_value = ([_region_instance(region="us-east-1")], [])
    result = CliRunner().invoke(main, ["status", "--region", "us-east-1", "--region", "eu-west-1"])
    assert result.exit_code == 0
    mock_regions.assert_not_called()
    assert mock_find.call_args[0][2] == ["us-east-1", "eu-west-1"]
    assert "Showing status for selected region(s): us-east-1, eu-west-1" in result.output


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1", "ap-south-1"])
def test_status_region_failure_warns_and_continues(
    mock_regions, mock_find, mock_session, mock_spot, mock_ssh_hosts, mock_details, mock_ebs
):
    """A failed region produces a warning but does not abort the command."""
    mock_find.return_value = (
        [_region_instance(region="us-east-1")],
        [{"region": "ap-south-1", "error": "AuthFailure: not authorized"}],
    )
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Skipped region ap-south-1" in result.output
    assert "i-east" in result.output


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1", "us-west-2"])
def test_status_shows_region_per_instance(
    mock_regions, mock_find, mock_session, mock_spot, mock_ssh_hosts, mock_details, mock_ebs
):
    """Each instance is labelled with its region, and the terminate hint pins it."""
    mock_find.return_value = (
        [_region_instance("i-east", "us-east-1"), _region_instance("i-west", "us-west-2")],
        [],
    )
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Region: us-east-1" in result.output
    assert "Region: us-west-2" in result.output
    assert "--region us-east-1" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1", "ap-south-1"])
def test_status_json_includes_regions_failed(mock_regions, mock_find, mock_session):
    """Structured output reports queried and failed regions."""
    mock_find.return_value = (
        [],
        [{"region": "ap-south-1", "error": "AuthFailure: not authorized"}],
    )
    result = CliRunner().invoke(main, ["-o", "json", "status"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["instances"] == []
    assert data["regions_queried"] == ["us-east-1", "ap-south-1"]
    assert data["regions_failed"] == [{"region": "ap-south-1", "error": "AuthFailure: not authorized"}]


def test_status_help_shows_region_repeatable():
    result = CliRunner().invoke(main, ["status", "--help"])
    assert result.exit_code == 0
    assert "repeatable" in result.output
    assert "all enabled regions" in result.output


# ---------------------------------------------------------------------------
# --gpu flag tests
# ---------------------------------------------------------------------------

_RUNNING_INSTANCE = {
    "InstanceId": "i-abc123",
    "Name": "aws-bootstrap-g4dn.xlarge",
    "State": "running",
    "InstanceType": "g4dn.xlarge",
    "PublicIp": "1.2.3.4",
    "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
    "Lifecycle": "spot",
    "AvailabilityZone": "us-west-2a",
}

_SAMPLE_GPU_INFO = GpuInfo(
    driver_version="560.35.03",
    cuda_driver_version="13.0",
    cuda_toolkit_version="12.8",
    gpu_name="Tesla T4",
    compute_capability="7.5",
    architecture="Turing",
)


def test_status_help_shows_gpu_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])
    assert result.exit_code == 0
    assert "--gpu" in result.output


@patch("aws_bootstrap.cli.query_gpu_info", return_value=_SAMPLE_GPU_INFO)
@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_gpu_shows_info(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu):
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2", "--gpu"])
    assert result.exit_code == 0
    assert "Tesla T4 (Turing)" in result.output
    assert "12.8" in result.output
    assert "driver supports up to 13.0" in result.output
    assert "560.35.03" in result.output
    mock_gpu.assert_called_once()


@patch("aws_bootstrap.cli.query_gpu_info", return_value=None)
@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_gpu_ssh_fails_gracefully(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu):
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2", "--gpu"])
    assert result.exit_code == 0
    assert "unavailable" in result.output


@patch("aws_bootstrap.cli.query_gpu_info", return_value=_SAMPLE_GPU_INFO)
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_gpu_no_ssh_config_uses_defaults(
    mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu
):
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2", "--gpu"])
    assert result.exit_code == 0
    # Should have been called with the instance IP and default user/key
    mock_gpu.assert_called_once()
    call_args = mock_gpu.call_args
    assert call_args[0][0] == "1.2.3.4"
    assert call_args[0][1] == "ubuntu"


@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_gpu_skips_non_running(mock_find, mock_session, mock_ssh_hosts, mock_details, mock_gpu):
    mock_find.return_value = [
        {
            "InstanceId": "i-stopped",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "stopped",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "on-demand",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2", "--gpu"])
    assert result.exit_code == 0
    mock_gpu.assert_not_called()


@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_without_gpu_flag_no_gpu_query(
    mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu
):
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    mock_gpu.assert_not_called()


# ---------------------------------------------------------------------------
# --instructions / --no-instructions / -I flag tests
# ---------------------------------------------------------------------------


def test_status_help_shows_instructions_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--help"])
    assert result.exit_code == 0
    assert "--instructions" in result.output
    assert "--no-instructions" in result.output
    assert "-I" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_instructions_shown_by_default(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    """Instructions are shown by default (no flag needed)."""
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "ssh aws-gpu1" in result.output
    assert "ssh -NL 8888:localhost:8888 aws-gpu1" in result.output
    assert "vscode-remote://ssh-remote+aws-gpu1/home/ubuntu/workspace" in result.output
    assert "python ~/gpu_benchmark.py" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_no_instructions_suppresses_commands(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    """--no-instructions suppresses connection commands."""
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2", "--no-instructions"])
    assert result.exit_code == 0
    assert "vscode-remote" not in result.output
    assert "Jupyter" not in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_instructions_no_alias_skips(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    """Instances without an SSH alias don't get connection instructions."""
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "ssh aws-gpu" not in result.output
    assert "vscode-remote" not in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_instructions_non_default_port(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519"), port=2222
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "ssh -p 2222 aws-gpu1" in result.output
    assert "ssh -NL 8888:localhost:8888 -p 2222 aws-gpu1" in result.output


# ---------------------------------------------------------------------------
# AWS credential / auth error handling tests
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.list_enabled_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_no_credentials_shows_friendly_error(mock_session, mock_regions):
    """NoCredentialsError should show a helpful message, not a raw traceback."""
    mock_regions.side_effect = botocore.exceptions.NoCredentialsError()
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "Unable to locate AWS credentials" in result.output
    assert "AWS_PROFILE" in result.output
    assert "--profile" in result.output
    assert "aws configure" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
def test_profile_not_found_shows_friendly_error(mock_session):
    """ProfileNotFound should show the missing profile name and list command."""
    mock_session.side_effect = botocore.exceptions.ProfileNotFound(profile="nonexistent")
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--profile", "nonexistent"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output
    assert "aws configure list-profiles" in result.output


@patch("aws_bootstrap.cli.list_enabled_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_partial_credentials_shows_friendly_error(mock_session, mock_regions):
    """PartialCredentialsError should mention incomplete credentials."""
    mock_regions.side_effect = botocore.exceptions.PartialCredentialsError(
        provider="env", cred_var="AWS_SECRET_ACCESS_KEY"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "Incomplete AWS credentials" in result.output
    assert "aws configure list" in result.output


@patch("aws_bootstrap.cli.list_enabled_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_expired_token_shows_friendly_error(mock_session, mock_regions):
    """ExpiredTokenException should show authorization failure with context."""
    mock_regions.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "ExpiredTokenException", "Message": "The security token is expired"}},
        "DescribeInstances",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "AWS authorization failed" in result.output
    assert "expired" in result.output.lower()


@patch("aws_bootstrap.cli.list_enabled_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_auth_failure_shows_friendly_error(mock_session, mock_regions):
    """AuthFailure ClientError should show authorization failure message."""
    mock_regions.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "AuthFailure", "Message": "credentials are invalid"}},
        "DescribeInstances",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "AWS authorization failed" in result.output


@patch("aws_bootstrap.cli.list_enabled_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_unhandled_client_error_propagates(mock_session, mock_regions):
    """Non-auth ClientErrors should propagate without being caught."""
    mock_regions.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "UnknownError", "Message": "something else"}},
        "DescribeInstances",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert isinstance(result.exception, botocore.exceptions.ClientError)


@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_no_credentials_caught_on_terminate(mock_session, mock_find):
    """Credential errors are caught for all subcommands, not just status."""
    mock_find.side_effect = botocore.exceptions.NoCredentialsError()
    runner = CliRunner()
    result = runner.invoke(main, ["terminate"])
    assert result.exit_code != 0
    assert "Unable to locate AWS credentials" in result.output


@patch("aws_bootstrap.cli.list_instance_types")
@patch("aws_bootstrap.cli.boto3.Session")
def test_no_credentials_caught_on_list(mock_session, mock_list):
    """Credential errors are caught for nested subcommands (list instance-types)."""
    mock_list.side_effect = botocore.exceptions.NoCredentialsError()
    runner = CliRunner()
    result = runner.invoke(main, ["list", "instance-types"])
    assert result.exit_code != 0
    assert "Unable to locate AWS credentials" in result.output


# ---------------------------------------------------------------------------
# --python-version tests
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_python_version_passed_to_setup(
    mock_session, mock_ami, mock_import, mock_sg, mock_launch, mock_wait, mock_ssh, mock_setup, mock_add_ssh, tmp_path
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = _region_launch({"InstanceId": "i-test123"})
    mock_wait.return_value = {
        "PublicIpAddress": "1.2.3.4",
        "Placement": {"AvailabilityZone": "us-west-2a"},
    }

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--python-version", "3.13"])
    assert result.exit_code == 0
    mock_setup.assert_called_once()
    assert mock_setup.call_args[0][4] == "3.13"


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_shows_python_version(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run", "--python-version", "3.14.2"])
    assert result.exit_code == 0
    assert "3.14.2" in result.output
    assert "Python version" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_omits_python_version_when_unset(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run"])
    assert result.exit_code == 0
    assert "Python version" not in result.output


# ---------------------------------------------------------------------------
# multi-region + --wait tests
# ---------------------------------------------------------------------------


def test_launch_help_shows_region_and_wait():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--help"])
    assert result.exit_code == 0
    assert "--wait" in result.output
    assert "--wait-timeout" in result.output
    # Help text is line-wrapped by Click; normalize whitespace before matching.
    flat = " ".join(result.output.split())
    assert "attempted one at a time in the given order" in flat
    assert "NOT one instance per region" in flat


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_json_multi_region_and_wait(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_session.return_value.region_name = None

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-o",
            "json",
            "launch",
            "--key-path",
            str(key_path),
            "--dry-run",
            "--region",
            "us-east-1",
            "--region",
            "us-west-2",
            "--wait",
            "--wait-timeout",
            "90s",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["regions"] == ["us-east-1", "us-west-2"]
    assert data["wait"] is True
    assert data["wait_timeout_seconds"] == 90


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_region_defaults_to_profile_region(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    """With no --region, the profile/env region is used (not hardcoded us-west-2)."""
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_session.return_value.region_name = "ap-south-1"

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path), "--dry-run"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["regions"] == ["ap-south-1"]


def test_launch_invalid_wait_timeout_rejected(tmp_path):
    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run", "--wait-timeout", "bogus"])
    assert result.exit_code != 0
    assert "Invalid duration" in result.output or "duration" in result.output.lower()


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_text_output_shows_resolved_region(
    mock_session, mock_launch, mock_wait, mock_ssh, mock_setup, mock_add_ssh, tmp_path
):
    mock_session.return_value.region_name = None
    mock_launch.return_value = _region_launch({"InstanceId": "i-xyz"}, region="us-east-1")
    mock_wait.return_value = {
        "PublicIpAddress": "1.2.3.4",
        "Placement": {"AvailabilityZone": "us-east-1a"},
    }

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(
        main, ["launch", "--key-path", str(key_path), "--no-setup", "--region", "us-west-2", "--region", "us-east-1"]
    )
    assert result.exit_code == 0
    assert "Region: us-east-1" in result.output


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_benchmark_hint_uses_venv_python(
    mock_session, mock_launch, mock_wait, mock_ssh, mock_setup, mock_add_ssh, tmp_path
):
    """`ssh alias 'cmd'` is non-interactive (no ~/.bashrc / venv activation)
    and Ubuntu 24.04 has no unversioned `python` — the GPU-benchmark hint
    must invoke the venv interpreter by absolute path."""
    mock_session.return_value.region_name = None
    mock_launch.return_value = _region_launch({"InstanceId": "i-xyz"}, region="us-west-2")
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2a"}}
    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    result = CliRunner().invoke(main, ["launch", "--key-path", str(key_path), "--no-setup"])
    assert result.exit_code == 0
    assert "~/venv/bin/python ~/gpu_benchmark.py" in result.output
    assert "'python ~/gpu_benchmark.py'" not in result.output


# ---------------------------------------------------------------------------
# --ssh-port tests
# ---------------------------------------------------------------------------


def test_launch_help_shows_ssh_port():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--help"])
    assert result.exit_code == 0
    assert "--ssh-port" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_shows_ssh_port_when_non_default(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run", "--ssh-port", "2222"])
    assert result.exit_code == 0
    assert "2222" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_omits_ssh_port_when_default(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run"])
    assert result.exit_code == 0
    assert "SSH port" not in result.output


# ---------------------------------------------------------------------------
# EBS data volume tests
# ---------------------------------------------------------------------------


def test_launch_help_shows_ebs_options():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--help"])
    assert result.exit_code == 0
    assert "--ebs-storage" in result.output
    assert "--ebs-volume-id" in result.output


def test_launch_ebs_mutual_exclusivity(tmp_path):
    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(
        main, ["launch", "--key-path", str(key_path), "--ebs-storage", "96", "--ebs-volume-id", "vol-abc"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_with_ebs_storage(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run", "--ebs-storage", "96"])
    assert result.exit_code == 0
    assert "96 GB gp3" in result.output
    assert "/data" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_dry_run_with_ebs_volume_id(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--dry-run", "--ebs-volume-id", "vol-abc"])
    assert result.exit_code == 0
    assert "vol-abc" in result.output
    assert "/data" in result.output


@patch("aws_bootstrap.cli.mount_ebs_volume", return_value=True)
@patch("aws_bootstrap.cli.attach_ebs_volume")
@patch("aws_bootstrap.cli.create_ebs_volume", return_value="vol-new123")
@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_with_ebs_storage_full_flow(
    mock_session,
    mock_ami,
    mock_import,
    mock_sg,
    mock_launch,
    mock_wait,
    mock_ssh,
    mock_setup,
    mock_add_ssh,
    mock_create_ebs,
    mock_attach_ebs,
    mock_mount_ebs,
    tmp_path,
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = _region_launch({"InstanceId": "i-test123"})
    mock_wait.return_value = {
        "PublicIpAddress": "1.2.3.4",
        "Placement": {"AvailabilityZone": "us-west-2a"},
    }

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--ebs-storage", "96", "--no-setup"])
    assert result.exit_code == 0
    assert "vol-new123" in result.output
    mock_create_ebs.assert_called_once()
    mock_attach_ebs.assert_called_once()
    mock_mount_ebs.assert_called_once()
    # Verify format_volume=True for new volumes
    assert mock_mount_ebs.call_args[1]["format_volume"] is True


@patch("aws_bootstrap.cli.mount_ebs_volume", return_value=True)
@patch("aws_bootstrap.cli.attach_ebs_volume")
@patch("aws_bootstrap.cli.validate_ebs_volume")
@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_with_ebs_volume_id_full_flow(
    mock_session,
    mock_ami,
    mock_import,
    mock_sg,
    mock_launch,
    mock_wait,
    mock_ssh,
    mock_setup,
    mock_add_ssh,
    mock_validate,
    mock_attach_ebs,
    mock_mount_ebs,
    tmp_path,
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = _region_launch({"InstanceId": "i-test123"})
    mock_wait.return_value = {
        "PublicIpAddress": "1.2.3.4",
        "Placement": {"AvailabilityZone": "us-west-2a"},
    }
    mock_validate.return_value = {"VolumeId": "vol-existing", "Size": 200}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(
        main, ["launch", "--key-path", str(key_path), "--ebs-volume-id", "vol-existing", "--no-setup"]
    )
    assert result.exit_code == 0
    mock_validate.assert_called_once()
    mock_attach_ebs.assert_called_once()
    mock_mount_ebs.assert_called_once()
    # Verify format_volume=False for existing volumes
    assert mock_mount_ebs.call_args[1]["format_volume"] is False


@patch("aws_bootstrap.cli.mount_ebs_volume", return_value=False)
@patch("aws_bootstrap.cli.attach_ebs_volume")
@patch("aws_bootstrap.cli.create_ebs_volume", return_value="vol-new123")
@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_ebs_mount_failure_warns(
    mock_session,
    mock_ami,
    mock_import,
    mock_sg,
    mock_launch,
    mock_wait,
    mock_ssh,
    mock_add_ssh,
    mock_create_ebs,
    mock_attach_ebs,
    mock_mount_ebs,
    tmp_path,
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = _region_launch({"InstanceId": "i-test123"})
    mock_wait.return_value = {
        "PublicIpAddress": "1.2.3.4",
        "Placement": {"AvailabilityZone": "us-west-2a"},
    }

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--ebs-storage", "96", "--no-setup"])
    # Should succeed despite mount failure (just a warning)
    assert result.exit_code == 0
    assert "WARNING" in result.output or "Failed to mount" in result.output


def test_terminate_help_shows_keep_ebs():
    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--help"])
    assert result.exit_code == 0
    assert "--keep-ebs" in result.output


@patch("aws_bootstrap.cli.delete_ebs_volume")
@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance")
@patch("aws_bootstrap.cli.remove_ssh_host", return_value=None)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
def test_terminate_deletes_ebs_by_default(
    mock_terminate, mock_find, mock_session, mock_remove_ssh, mock_find_ebs, mock_delete_ebs
):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    mock_find_ebs.return_value = [{"VolumeId": "vol-data1", "Size": 96, "Device": "/dev/sdf", "State": "in-use"}]

    # Mock the ec2 client's get_waiter for volume_available
    mock_ec2 = mock_session.return_value.client.return_value
    mock_waiter = MagicMock()
    mock_ec2.get_waiter.return_value = mock_waiter

    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes"])
    assert result.exit_code == 0
    mock_delete_ebs.assert_called_once_with(mock_ec2, "vol-data1")


@patch("aws_bootstrap.cli.delete_ebs_volume")
@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance")
@patch("aws_bootstrap.cli.remove_ssh_host", return_value=None)
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
def test_terminate_keep_ebs_preserves(
    mock_terminate, mock_find, mock_session, mock_remove_ssh, mock_find_ebs, mock_delete_ebs
):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    mock_find_ebs.return_value = [{"VolumeId": "vol-data1", "Size": 96, "Device": "/dev/sdf", "State": "in-use"}]

    runner = CliRunner()
    result = runner.invoke(main, ["terminate", "--yes", "--keep-ebs"])
    assert result.exit_code == 0
    assert "Preserving EBS volume: vol-data1" in result.output
    assert "aws-bootstrap launch --ebs-volume-id vol-data1" in result.output
    mock_delete_ebs.assert_not_called()


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance")
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_shows_ebs_volumes(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details, mock_ebs):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.15
    mock_ebs.return_value = [{"VolumeId": "vol-data1", "Size": 96, "Device": "/dev/sdf", "State": "in-use"}]

    runner = CliRunner()
    result = runner.invoke(main, ["status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "vol-data1" in result.output
    assert "96 GB" in result.output
    assert "/data" in result.output


# ---------------------------------------------------------------------------
# cleanup subcommand
# ---------------------------------------------------------------------------


def _writekey(tmp_path):
    k = tmp_path / "id_ed25519.pub"
    k.write_text("ssh-ed25519 AAAA t@h")
    return k


def _live(instance_id, public_ip="1.2.3.4", region="us-east-1", cluster_id="", rank=None):
    return {"InstanceId": instance_id, "PublicIp": public_ip, "Region": region, "ClusterId": cluster_id, "Rank": rank}


def test_cleanup_help():
    runner = CliRunner()
    result = runner.invoke(main, ["cleanup", "--help"])
    assert result.exit_code == 0
    for opt in ("--dry-run", "--yes", "--sync", "--key-path"):
        assert opt in result.output
    assert "--region" not in result.output  # cleanup is always all-regions now


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions", return_value=([], []))
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_nothing_to_do(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["cleanup"])
    assert result.exit_code == 0
    assert "in sync" in result.output


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1", "us-west-2"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_keeps_alive_instance_in_other_region(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    add_ssh_host("i-east0001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg)
    mock_find.return_value = ([_live("i-east0001", "1.1.1.1", "us-east-1")], [])
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["cleaned"] == []
    assert "i-east0001" in cfg.read_text()


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_removes_stale(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    add_ssh_host("i-dead0001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg)
    mock_find.return_value = ([], [])  # not alive anywhere, no failures
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [c["instance_id"] for c in data["cleaned"]] == ["i-dead0001"]
    assert "i-dead0001" not in cfg.read_text()


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_dry_run_does_not_modify(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    add_ssh_host("i-dead0001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg)
    mock_find.return_value = ([], [])
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["cleanup", "--dry-run"])
    assert result.exit_code == 0
    assert "would remove" in result.output
    assert "i-dead0001" in cfg.read_text()  # unchanged


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_conservative_guard_skips_removal_on_region_failure(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    add_ssh_host("i-gone0001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg)
    # Not in the (partial) live list, but a region query FAILED -> must not remove.
    mock_find.return_value = ([], [{"region": "us-east-1", "error": "AuthFailure"}])
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["cleaned"] == []
    assert data["regions_failed"]
    assert "i-gone0001" in cfg.read_text()


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-ml1-0")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_sync_adds_missing_cluster_alias(mock_session, mock_find, mock_regions, mock_add, tmp_path):
    cfg = tmp_path / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    mock_find.return_value = ([_live("i-c1000001", "1.2.3.4", cluster_id="ml1", rank=0)], [])
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(
            main, ["-o", "json", "cleanup", "--sync", "--yes", "--key-path", str(_writekey(tmp_path))]
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert any(a["instance_id"] == "i-c1000001" for a in data["added"])
    assert mock_add.call_args.kwargs.get("alias") == "aws-ml1-0"  # deterministic cluster alias


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_sync_repairs_drift(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    add_ssh_host("i-d41f7001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg)  # old IP
    mock_find.return_value = ([_live("i-d41f7001", "9.9.9.9")], [])  # new IP
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(
            main, ["-o", "json", "cleanup", "--sync", "--yes", "--key-path", str(_writekey(tmp_path))]
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [u["instance_id"] for u in data["updated"]] == ["i-d41f7001"]
    assert "9.9.9.9" in cfg.read_text()


def test_cleanup_requires_yes_in_structured():
    with patch("aws_bootstrap.cli.boto3.Session"):
        result = CliRunner().invoke(main, ["-o", "json", "cleanup"])
    assert result.exit_code != 0


@pytest.mark.parametrize("fmt", ["json", "yaml", "table"])
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_renders_in_each_structured_format(mock_session, mock_find, mock_regions, tmp_path, fmt):
    cfg = tmp_path / "config"
    add_ssh_host("i-dead0001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg)
    mock_find.return_value = ([], [])
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["-o", fmt, "cleanup", "--yes"])
    assert result.exit_code == 0, result.output
    assert "i-dead0001" in result.output  # the removed alias's instance appears in output


# ---------------------------------------------------------------------------
# cleanup --include-ebs (orphan volumes, scanned across regions)
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.delete_ebs_volume")
@patch("aws_bootstrap.cli.find_orphan_ebs_volumes")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions", return_value=([], []))
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_include_ebs_deletes_orphans(mock_session, mock_find, mock_regions, mock_orphan, mock_delete, tmp_path):
    cfg = tmp_path / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    mock_orphan.return_value = [{"VolumeId": "vol-orphan1", "Size": 50, "State": "available", "InstanceId": "i-x"}]
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["-o", "json", "cleanup", "--include-ebs", "--yes"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["deleted_volumes"][0]["volume_id"] == "vol-orphan1"
    mock_delete.assert_called_once()


@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_sync_drift_preserves_port(mock_session, mock_find, mock_regions, tmp_path):
    cfg = tmp_path / "config"
    add_ssh_host("i-0aaf7001", "1.1.1.1", "ubuntu", _writekey(tmp_path), config_path=cfg, port=2222)
    mock_find.return_value = ([_live("i-0aaf7001", "9.9.9.9")], [])
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["cleanup", "--sync", "--yes", "--key-path", str(_writekey(tmp_path))])
    assert result.exit_code == 0, result.output
    content = cfg.read_text()
    assert "HostName 9.9.9.9" in content  # IP repaired
    assert "Port 2222" in content  # non-standard port preserved, not reset to 22


@patch("aws_bootstrap.cli.delete_ebs_volume")
@patch("aws_bootstrap.cli.find_orphan_ebs_volumes")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_include_ebs_not_deleted_on_region_failure(
    mock_session, mock_find, mock_regions, mock_orphan, mock_delete, tmp_path
):
    cfg = tmp_path / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    mock_find.return_value = ([], [{"region": "us-east-1", "error": "AuthFailure"}])  # incomplete scan
    mock_orphan.return_value = [{"VolumeId": "vol-x", "Size": 50, "State": "available", "InstanceId": "i-x"}]
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["-o", "json", "cleanup", "--include-ebs", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    mock_delete.assert_not_called()  # conservative guard: don't delete on incomplete scan
    assert "orphan_volumes" in data  # reported as candidates, not deleted_volumes


@patch("aws_bootstrap.cli.find_orphan_ebs_volumes")
@patch("aws_bootstrap.cli.list_enabled_regions", return_value=["us-east-1"])
@patch("aws_bootstrap.cli.find_tagged_instances_in_regions", return_value=([], []))
@patch("aws_bootstrap.cli.boto3.Session")
def test_cleanup_without_include_ebs_skips_volume_check(mock_session, mock_find, mock_regions, mock_orphan, tmp_path):
    cfg = tmp_path / "config"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("")
    with patch("aws_bootstrap.cli._SSH_CONFIG_PATH", cfg):
        result = CliRunner().invoke(main, ["cleanup"])
    assert result.exit_code == 0
    mock_orphan.assert_not_called()


# ---------------------------------------------------------------------------
# --output structured format tests
# ---------------------------------------------------------------------------


def test_help_shows_output_option():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "-o" in result.output


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_output_json(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details, mock_ebs):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.1578
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "status", "--region", "us-west-2"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "instances" in data
    assert data["regions_queried"] == ["us-west-2"]
    assert len(data["instances"]) == 1
    inst = data["instances"][0]
    assert inst["instance_id"] == "i-abc123"
    assert inst["region"] == "us-west-2"
    assert inst["state"] == "running"
    assert inst["instance_type"] == "g4dn.xlarge"
    assert inst["public_ip"] == "1.2.3.4"
    assert inst["lifecycle"] == "spot"
    assert inst["spot_price_per_hour"] == 0.1578
    assert "uptime_seconds" in inst
    assert "estimated_cost" in inst
    # No ANSI or progress text in structured output
    assert "\x1b[" not in result.output
    assert "Found" not in result.output


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_output_yaml(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details, mock_ebs):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.15
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "yaml", "status", "--region", "us-west-2"])
    assert result.exit_code == 0
    data = yaml.safe_load(result.output)
    assert "instances" in data
    assert data["instances"][0]["instance_id"] == "i-abc123"


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_output_table(mock_find, mock_spot_price, mock_session, mock_ssh_hosts, mock_details, mock_ebs):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
            "Lifecycle": "spot",
            "AvailabilityZone": "us-west-2a",
        }
    ]
    mock_spot_price.return_value = 0.15
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "table", "status", "--region", "us-west-2"])
    assert result.exit_code == 0
    assert "Instance ID" in result.output
    assert "i-abc123" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.ec2.find_tagged_instances")
def test_status_no_instances_json(mock_find, mock_session):
    mock_find.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "status", "--region", "us-west-2"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {"instances": [], "regions_queried": ["us-west-2"]}


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
def test_launch_output_json_dry_run(mock_sg, mock_import, mock_ami, mock_session, tmp_path):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}

    key_path = tmp_path / "id_ed25519.pub"
    key_path.write_text("ssh-ed25519 AAAA test@host")

    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path), "--dry-run"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["dry_run"] is True
    assert data["instance_type"] == "g4dn.xlarge"
    assert data["ami_id"] == "ami-123"
    assert data["pricing"] == "spot"
    assert data["regions"] == ["us-west-2"]
    assert data["wait"] is False
    assert data["wait_timeout_seconds"] == 1800


@patch("aws_bootstrap.cli.remove_ssh_host", return_value="aws-gpu1")
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
def test_terminate_output_json(mock_terminate, mock_find, mock_session, mock_remove_ssh):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    mock_terminate.return_value = [
        {
            "InstanceId": "i-abc123",
            "PreviousState": {"Name": "running"},
            "CurrentState": {"Name": "shutting-down"},
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "terminate", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "terminated" in data
    assert len(data["terminated"]) == 1
    assert data["terminated"][0]["instance_id"] == "i-abc123"
    assert data["terminated"][0]["previous_state"] == "running"
    assert data["terminated"][0]["current_state"] == "shutting-down"
    assert data["terminated"][0]["ssh_alias_removed"] == "aws-gpu1"


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.list_instance_types")
def test_list_instance_types_json(mock_list, mock_session):
    mock_list.return_value = [
        {
            "InstanceType": "g4dn.xlarge",
            "VCpuCount": 4,
            "MemoryMiB": 16384,
            "GpuSummary": "1x T4 (16384 MiB)",
        },
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "list", "instance-types"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["instance_type"] == "g4dn.xlarge"
    assert data[0]["vcpus"] == 4
    assert data[0]["memory_mib"] == 16384
    assert data[0]["gpu"] == "1x T4 (16384 MiB)"


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.list_amis")
def test_list_amis_json(mock_list, mock_session):
    mock_list.return_value = [
        {
            "ImageId": "ami-abc123",
            "Name": "Deep Learning AMI v42",
            "CreationDate": "2025-06-01T00:00:00Z",
            "Architecture": "x86_64",
        },
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "list", "amis"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["image_id"] == "ami-abc123"
    assert data[0]["name"] == "Deep Learning AMI v42"
    assert data[0]["creation_date"] == "2025-06-01"


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_terminate_json_requires_yes(mock_find, mock_session):
    """Structured output without --yes should error."""
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "test",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "terminate"])
    assert result.exit_code != 0
    assert "--yes is required" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_terminate_no_instances_json(mock_find, mock_session):
    mock_find.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "terminate", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {"terminated": []}


# ---------------------------------------------------------------------------
# quota subcommand
# ---------------------------------------------------------------------------

_SPOT_QUOTA = {
    "quota_code": "L-3819A6DF",
    "quota_name": "All G and VT Spot Instance Requests",
    "value": 4.0,
    "quota_type": "spot",
    "family": "gvt",
}

_ON_DEMAND_QUOTA = {
    "quota_code": "L-DB2E81BA",
    "quota_name": "Running On-Demand G and VT instances",
    "value": 0.0,
    "quota_type": "on-demand",
    "family": "gvt",
}


def test_quota_help():
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "request" in result.output
    assert "history" in result.output


def test_quota_show_help():
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "show", "--help"])
    assert result.exit_code == 0
    assert "--region" in result.output
    assert "--profile" in result.output
    assert "--family" in result.output


def test_quota_request_help():
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "request", "--help"])
    assert result.exit_code == 0
    assert "--type" in result.output
    assert "--desired-value" in result.output
    assert "--yes" in result.output
    assert "--family" in result.output


def test_quota_history_help():
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "history", "--help"])
    assert result.exit_code == 0
    assert "--type" in result.output
    assert "--status" in result.output
    assert "--family" in result.output


@patch("aws_bootstrap.cli.get_family_quotas")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_show_text(mock_session, mock_quotas):
    mock_quotas.return_value = [_SPOT_QUOTA, _ON_DEMAND_QUOTA]
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "show", "--family", "gvt"])
    assert result.exit_code == 0
    assert "4" in result.output
    assert "Spot" in result.output
    assert "On-demand" in result.output


@patch("aws_bootstrap.cli.get_family_quotas")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_show_json(mock_session, mock_quotas):
    mock_quotas.return_value = [_SPOT_QUOTA, _ON_DEMAND_QUOTA]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "quota", "show", "--family", "gvt"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "quotas" in data
    assert len(data["quotas"]) == 2
    assert data["quotas"][0]["quota_code"] == "L-3819A6DF"
    assert data["quotas"][0]["value"] == 4.0


@patch("aws_bootstrap.cli.get_family_quotas")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_show_api_error(mock_session, mock_quotas):
    mock_quotas.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "NoSuchResourceException", "Message": "not found"}},
        "GetServiceQuota",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "show", "--family", "gvt"])
    assert result.exit_code != 0


@patch("aws_bootstrap.cli.request_quota_increase")
@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_with_yes(mock_session, mock_get, mock_request):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0}
    mock_request.return_value = {
        "request_id": "req-123",
        "status": "PENDING",
        "quota_code": "L-3819A6DF",
        "quota_name": "Spot Quota",
        "desired_value": 4.0,
    }
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "request", "--type", "spot", "--desired-value", "4", "--yes"])
    assert result.exit_code == 0
    assert "submitted" in result.output
    assert "req-123" in result.output
    mock_request.assert_called_once()


@patch("aws_bootstrap.cli.request_quota_increase")
@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_with_confirm(mock_session, mock_get, mock_request):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0}
    mock_request.return_value = {
        "request_id": "req-123",
        "status": "PENDING",
        "quota_code": "L-3819A6DF",
        "quota_name": "Spot Quota",
        "desired_value": 4.0,
    }
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "request", "--type", "spot", "--desired-value", "4"], input="y\n")
    assert result.exit_code == 0
    assert "submitted" in result.output
    mock_request.assert_called_once()


@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_cancelled(mock_session, mock_get):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0}
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "request", "--type", "spot", "--desired-value", "4"], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output


@patch("aws_bootstrap.cli.request_quota_increase")
@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_json(mock_session, mock_get, mock_request):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0}
    mock_request.return_value = {
        "request_id": "req-123",
        "status": "PENDING",
        "quota_code": "L-3819A6DF",
        "quota_name": "Spot Quota",
        "desired_value": 4.0,
    }
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "quota", "request", "--type", "spot", "--desired-value", "4", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["requests"]) == 1
    assert data["requests"][0]["request_id"] == "req-123"
    assert data["requests"][0]["status"] == "PENDING"
    assert data["requests"][0]["region"] == "us-west-2"


@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_json_requires_yes(mock_session, mock_get):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0}
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "quota", "request", "--type", "spot", "--desired-value", "4"])
    assert result.exit_code != 0
    assert "--yes is required" in result.output


@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_desired_le_current(mock_session, mock_get):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 4.0}
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "request", "--type", "spot", "--desired-value", "4", "--yes"])
    assert result.exit_code != 0
    assert "must be greater" in result.output


@patch("aws_bootstrap.cli.request_quota_increase")
@patch("aws_bootstrap.cli.get_quota")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_request_duplicate_pending(mock_session, mock_get, mock_request):
    mock_get.return_value = {"quota_code": "L-3819A6DF", "quota_name": "Spot Quota", "value": 0.0}
    mock_request.side_effect = CLIError("already pending")
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "request", "--type", "spot", "--desired-value", "4", "--yes"])
    assert result.exit_code != 0
    assert "already pending" in result.output


@patch("aws_bootstrap.cli.get_quota_request_history")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_history_text(mock_session, mock_history):
    mock_history.return_value = [
        {
            "request_id": "req-123",
            "status": "APPROVED",
            "quota_code": "L-3819A6DF",
            "quota_name": "Spot Quota",
            "desired_value": 4.0,
            "created": "2025-06-01T00:00:00Z",
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "history", "--family", "gvt", "--type", "spot"])
    assert result.exit_code == 0
    assert "req-123" in result.output
    assert "APPROVED" in result.output


@patch("aws_bootstrap.cli.get_quota_request_history")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_history_json(mock_session, mock_history):
    mock_history.return_value = [
        {
            "request_id": "req-123",
            "status": "APPROVED",
            "quota_code": "L-3819A6DF",
            "quota_name": "Spot Quota",
            "desired_value": 4.0,
            "created": "2025-06-01T00:00:00Z",
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "quota", "history", "--family", "gvt", "--type", "spot"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "requests" in data
    assert len(data["requests"]) == 1
    assert data["requests"][0]["request_id"] == "req-123"


@patch("aws_bootstrap.cli.get_quota_request_history")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_history_all_families(mock_session, mock_history):
    """Without --type or --family, queries all families x all types."""
    mock_history.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "history"])
    assert result.exit_code == 0
    # Should be called 6 times (3 families x 2 types each)
    assert mock_history.call_count == 6


@patch("aws_bootstrap.cli.get_quota_request_history")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_history_filter_by_status(mock_session, mock_history):
    mock_history.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "history", "--family", "gvt", "--type", "spot", "--status", "APPROVED"])
    assert result.exit_code == 0
    mock_history.assert_called_once()
    call_kwargs = mock_history.call_args[1]
    assert call_kwargs["status_filter"] == "APPROVED"


@patch("aws_bootstrap.cli.get_quota_request_history")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_history_empty_text(mock_session, mock_history):
    mock_history.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["quota", "history", "--family", "gvt", "--type", "spot"])
    assert result.exit_code == 0
    assert "No quota increase requests found" in result.output


@patch("aws_bootstrap.cli.get_quota_request_history")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_history_empty_json(mock_session, mock_history):
    mock_history.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "quota", "history", "--family", "gvt", "--type", "spot"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {"requests": []}


@patch("aws_bootstrap.cli.get_family_quotas")
@patch("aws_bootstrap.cli.boto3.Session")
def test_quota_show_all_families(mock_session, mock_quotas):
    """Without --family, shows all families (gvt, p, dl)."""
    mock_quotas.side_effect = [
        [_SPOT_QUOTA, _ON_DEMAND_QUOTA],
        [
            {"quota_code": "L-7212CCBC", "quota_name": "P Spot", "value": 0.0, "quota_type": "spot", "family": "p"},
            {
                "quota_code": "L-417A185B",
                "quota_name": "P On-Demand",
                "value": 0.0,
                "quota_type": "on-demand",
                "family": "p",
            },
        ],
        [
            {"quota_code": "L-85EED4F7", "quota_name": "DL Spot", "value": 0.0, "quota_type": "spot", "family": "dl"},
            {
                "quota_code": "L-6E869C2A",
                "quota_name": "DL On-Demand",
                "value": 0.0,
                "quota_type": "on-demand",
                "family": "dl",
            },
        ],
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "quota", "show"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["quotas"]) == 6
    assert mock_quotas.call_count == 3
