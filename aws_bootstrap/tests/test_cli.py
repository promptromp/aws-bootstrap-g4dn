"""Tests for CLI entry point and help output."""

from __future__ import annotations
from datetime import UTC, datetime
from unittest.mock import patch

from click.testing import CliRunner

from aws_bootstrap.cli import main


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
    assert "0.1.0" in result.output


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


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_shows_instances(mock_find, mock_spot_price, mock_session):
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


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.get_spot_price")
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_on_demand_no_cost(mock_find, mock_spot_price, mock_session):
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


@patch("aws_bootstrap.cli.boto3.Session")
@patch("aws_bootstrap.cli.find_tagged_instances")
@patch("aws_bootstrap.cli.terminate_tagged_instances")
def test_terminate_with_confirm(mock_terminate, mock_find, mock_session):
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
