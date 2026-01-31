"""Tests for CLI entry point and help output."""

from __future__ import annotations
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.ssh import GpuInfo, SSHHostDetails


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


@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_shows_instances(mock_find, mock_spot_price, mock_session, mock_ssh_hosts):
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


@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_on_demand_no_cost(mock_find, mock_spot_price, mock_session, mock_ssh_hosts):
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


@patch("aws_bootstrap.cli.list_ssh_hosts")
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_shows_alias(mock_find, mock_spot_price, mock_session, mock_ssh_hosts):
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


@patch("aws_bootstrap.cli.list_ssh_hosts", return_value={})
@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_no_alias_graceful(mock_find, mock_spot_price, mock_session, mock_ssh_hosts):
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
def test_status_without_gpu_flag_no_ssh(mock_find, mock_spot, mock_session, mock_ssh_hosts, mock_details, mock_gpu):
    mock_find.return_value = [_RUNNING_INSTANCE]
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    mock_gpu.assert_not_called()
    mock_details.assert_not_called()
