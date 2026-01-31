"""Tests for CLI entry point and help output."""

from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import botocore.exceptions
from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.gpu import GpuInfo
from aws_bootstrap.ssh import SSHHostDetails


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
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4"}

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
    assert "vscode-remote://ssh-remote+aws-gpu1/home/ubuntu" in result.output
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
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4"}

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
