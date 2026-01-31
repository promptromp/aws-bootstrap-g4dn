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
@patch("aws_bootstrap.cli.find_tagged_instances")
def test_status_shows_instances(mock_find, mock_session):
    mock_find.return_value = [
        {
            "InstanceId": "i-abc123",
            "Name": "aws-bootstrap-g4dn.xlarge",
            "State": "running",
            "InstanceType": "g4dn.xlarge",
            "PublicIp": "1.2.3.4",
            "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        }
    ]
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "i-abc123" in result.output
    assert "1.2.3.4" in result.output


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
