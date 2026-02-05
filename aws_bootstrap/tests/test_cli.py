"""Tests for CLI entry point and help output."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import botocore.exceptions
import yaml
from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.gpu import GpuInfo
from aws_bootstrap.ssh import CleanupResult, SSHHostDetails


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


def test_launch_missing_key():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--key-path", "/nonexistent/key.pub"])
    assert result.exit_code != 0
    assert "SSH public key not found" in result.output


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
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_no_instances(mock_find, mock_session):
    mock_find.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "No active" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["status"])
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
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["status"])
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
@patch("aws_bootstrap.cli.launch_instance")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_output_shows_ssh_alias(
    mock_session, mock_ami, mock_import, mock_sg, mock_launch, mock_wait, mock_ssh, mock_setup, mock_add_ssh, tmp_path
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = {"InstanceId": "i-test123"}
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
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "aws-gpu1" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "i-old999" in result.output


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
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_gpu_shows_info(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu):
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--gpu"])
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
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_gpu_ssh_fails_gracefully(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu):
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--gpu"])
    assert result.exit_code == 0
    assert "unavailable" in result.output


@patch("aws_bootstrap.cli.query_gpu_info", return_value=_SAMPLE_GPU_INFO)
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_gpu_no_ssh_config_uses_defaults(
    mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu
):
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--gpu"])
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
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["status", "--gpu"])
    assert result.exit_code == 0
    mock_gpu.assert_not_called()


@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_without_gpu_flag_no_gpu_query(
    mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu
):
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
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
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_instructions_shown_by_default(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    """Instructions are shown by default (no flag needed)."""
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "ssh aws-gpu1" in result.output
    assert "ssh -NL 8888:localhost:8888 aws-gpu1" in result.output
    assert "vscode-remote://ssh-remote+aws-gpu1/home/ubuntu/workspace" in result.output
    assert "python ~/gpu_benchmark.py" in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_no_instructions_suppresses_commands(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    """--no-instructions suppresses connection commands."""
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519")
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status", "--no-instructions"])
    assert result.exit_code == 0
    assert "vscode-remote" not in result.output
    assert "Jupyter" not in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_instructions_no_alias_skips(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    """Instances without an SSH alias don't get connection instructions."""
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "ssh aws-gpu" not in result.output
    assert "vscode-remote" not in result.output


@patch("aws_bootstrap.cli.get_ssh_host_details")
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={"i-abc123": "aws-gpu1"})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price", return_value=0.15)
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_instructions_non_default_port(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details):
    mock_find.return_value = [_RUNNING_INSTANCE]
    mock_details.return_value = SSHHostDetails(
        hostname="1.2.3.4", user="ubuntu", identity_file=Path("/home/user/.ssh/id_ed25519"), port=2222
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "ssh -p 2222 aws-gpu1" in result.output
    assert "ssh -NL 8888:localhost:8888 -p 2222 aws-gpu1" in result.output


# ---------------------------------------------------------------------------
# AWS credential / auth error handling tests
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_no_credentials_shows_friendly_error(mock_session, mock_find):
    """NoCredentialsError should show a helpful message, not a raw traceback."""
    mock_find.side_effect = botocore.exceptions.NoCredentialsError()
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


@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_partial_credentials_shows_friendly_error(mock_session, mock_find):
    """PartialCredentialsError should mention incomplete credentials."""
    mock_find.side_effect = botocore.exceptions.PartialCredentialsError(
        provider="env", cred_var="AWS_SECRET_ACCESS_KEY"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "Incomplete AWS credentials" in result.output
    assert "aws configure list" in result.output


@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_expired_token_shows_friendly_error(mock_session, mock_find):
    """ExpiredTokenException should show authorization failure with context."""
    mock_find.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "ExpiredTokenException", "Message": "The security token is expired"}},
        "DescribeInstances",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "AWS authorization failed" in result.output
    assert "expired" in result.output.lower()


@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_auth_failure_shows_friendly_error(mock_session, mock_find):
    """AuthFailure ClientError should show authorization failure message."""
    mock_find.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "AuthFailure", "Message": "credentials are invalid"}},
        "DescribeInstances",
    )
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0
    assert "AWS authorization failed" in result.output


@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_unhandled_client_error_propagates(mock_session, mock_find):
    """Non-auth ClientErrors should propagate without being caught."""
    mock_find.side_effect = botocore.exceptions.ClientError(
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
@patch("aws_bootstrap.cli.launch_instance")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami")
@patch("aws_bootstrap.cli.boto3.Session")
def test_launch_python_version_passed_to_setup(
    mock_session, mock_ami, mock_import, mock_sg, mock_launch, mock_wait, mock_ssh, mock_setup, mock_add_ssh, tmp_path
):
    mock_ami.return_value = {"ImageId": "ami-123", "Name": "TestAMI"}
    mock_launch.return_value = {"InstanceId": "i-test123"}
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
@patch("aws_bootstrap.cli.launch_instance")
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
    mock_launch.return_value = {"InstanceId": "i-test123"}
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
@patch("aws_bootstrap.cli.launch_instance")
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
    mock_launch.return_value = {"InstanceId": "i-test123"}
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
@patch("aws_bootstrap.cli.launch_instance")
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
    mock_launch.return_value = {"InstanceId": "i-test123"}
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
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "vol-data1" in result.output
    assert "96 GB" in result.output
    assert "/data" in result.output


# ---------------------------------------------------------------------------
# cleanup subcommand
# ---------------------------------------------------------------------------


def test_cleanup_help():
    runner = CliRunner()
    result = runner.invoke(main, ["cleanup", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--yes" in result.output
    assert "--region" in result.output
    assert "--profile" in result.output


@patch("aws_bootstrap.cli.find_stale_ssh_hosts", return_value=[])
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances", return_value=[])
def test_cleanup_no_stale(mock_find, mock_session, mock_stale):
    runner = CliRunner()
    result = runner.invoke(main, ["cleanup"])
    assert result.exit_code == 0
    assert "No stale" in result.output


@patch("aws_bootstrap.cli.find_stale_ssh_hosts", return_value=[("i-dead1234", "aws-gpu1")])
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances", return_value=[])
def test_cleanup_dry_run(mock_find, mock_session, mock_stale):
    runner = CliRunner()
    result = runner.invoke(main, ["cleanup", "--dry-run"])
    assert result.exit_code == 0
    assert "Would remove" in result.output
    assert "aws-gpu1" in result.output
    assert "i-dead1234" in result.output


@patch("aws_bootstrap.cli.cleanup_stale_ssh_hosts")
@patch("aws_bootstrap.cli.find_stale_ssh_hosts", return_value=[("i-dead1234", "aws-gpu1")])
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances", return_value=[])
def test_cleanup_with_yes(mock_find, mock_session, mock_stale, mock_cleanup):
    mock_cleanup.return_value = [CleanupResult(instance_id="i-dead1234", alias="aws-gpu1", removed=True)]
    runner = CliRunner()
    result = runner.invoke(main, ["cleanup", "--yes"])
    assert result.exit_code == 0
    assert "Removed aws-gpu1" in result.output
    assert "Cleaned up 1" in result.output
    mock_cleanup.assert_called_once()


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
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["-o", "json", "status"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "instances" in data
    assert len(data["instances"]) == 1
    inst = data["instances"][0]
    assert inst["instance_id"] == "i-abc123"
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
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["-o", "yaml", "status"])
    assert result.exit_code == 0
    data = yaml.safe_load(result.output)
    assert "instances" in data
    assert data["instances"][0]["instance_id"] == "i-abc123"


@patch("aws_bootstrap.cli.find_ebs_volumes_for_instance", return_value=[])
@patch("aws_bootstrap.cli.get_ssh_host_details", return_value=None)
@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
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
    result = runner.invoke(main, ["-o", "table", "status"])
    assert result.exit_code == 0
    assert "Instance ID" in result.output
    assert "i-abc123" in result.output


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_no_instances_json(mock_find, mock_session):
    mock_find.return_value = []
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "status"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {"instances": []}


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
    assert data["region"] == "us-west-2"


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


@patch("aws_bootstrap.cli.cleanup_stale_ssh_hosts")
@patch("aws_bootstrap.cli.find_stale_ssh_hosts", return_value=[("i-dead1234", "aws-gpu1")])
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances", return_value=[])
def test_cleanup_output_json(mock_find, mock_session, mock_stale, mock_cleanup):
    mock_cleanup.return_value = [CleanupResult(instance_id="i-dead1234", alias="aws-gpu1", removed=True)]
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "cleanup", "--yes"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "cleaned" in data
    assert len(data["cleaned"]) == 1
    assert data["cleaned"][0]["instance_id"] == "i-dead1234"
    assert data["cleaned"][0]["alias"] == "aws-gpu1"
    assert data["cleaned"][0]["removed"] is True


@patch("aws_bootstrap.cli.find_stale_ssh_hosts", return_value=[("i-dead1234", "aws-gpu1")])
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances", return_value=[])
def test_cleanup_dry_run_json(mock_find, mock_session, mock_stale):
    runner = CliRunner()
    result = runner.invoke(main, ["-o", "json", "cleanup", "--dry-run"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["dry_run"] is True
    assert "stale" in data
    assert data["stale"][0]["alias"] == "aws-gpu1"


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
