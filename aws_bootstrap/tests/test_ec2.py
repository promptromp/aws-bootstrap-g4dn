"""Tests for EC2 helper functions."""

from __future__ import annotations
from datetime import UTC, datetime
from unittest.mock import MagicMock

import botocore.exceptions
import click
import pytest

from aws_bootstrap.config import LaunchConfig
from aws_bootstrap.ec2 import (
    find_tagged_instances,
    get_latest_ami,
    launch_instance,
    terminate_tagged_instances,
)


def test_get_latest_ami_picks_newest():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {
        "Images": [
            {"ImageId": "ami-old", "Name": "DL AMI old", "CreationDate": "2024-01-01T00:00:00Z"},
            {"ImageId": "ami-new", "Name": "DL AMI new", "CreationDate": "2025-06-01T00:00:00Z"},
            {"ImageId": "ami-mid", "Name": "DL AMI mid", "CreationDate": "2025-01-01T00:00:00Z"},
        ]
    }
    ami = get_latest_ami(ec2, "DL AMI*")
    assert ami["ImageId"] == "ami-new"


def test_get_latest_ami_no_results():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {"Images": []}
    with pytest.raises(click.ClickException, match="No AMI found"):
        get_latest_ami(ec2, "nonexistent*")


def _make_client_error(code: str, message: str = "test") -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError(
        {"Error": {"Code": code, "Message": message}},
        "RunInstances",
    )


def test_launch_instance_spot_quota_exceeded():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("MaxSpotInstanceCountExceeded")
    config = LaunchConfig(spot=True)
    with pytest.raises(click.ClickException, match="Spot instance quota exceeded"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_launch_instance_vcpu_limit_exceeded():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("VcpuLimitExceeded")
    config = LaunchConfig(spot=False)
    with pytest.raises(click.ClickException, match="vCPU quota exceeded"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_launch_instance_quota_error_includes_readme_hint():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("MaxSpotInstanceCountExceeded")
    config = LaunchConfig(spot=True)
    with pytest.raises(click.ClickException, match="README.md"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_find_tagged_instances():
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-abc123",
                        "State": {"Name": "running"},
                        "InstanceType": "g4dn.xlarge",
                        "PublicIpAddress": "1.2.3.4",
                        "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
                        "Tags": [
                            {"Key": "Name", "Value": "aws-bootstrap-g4dn.xlarge"},
                            {"Key": "created-by", "Value": "aws-bootstrap-g4dn"},
                        ],
                    }
                ]
            }
        ]
    }
    instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    assert len(instances) == 1
    assert instances[0]["InstanceId"] == "i-abc123"
    assert instances[0]["State"] == "running"
    assert instances[0]["PublicIp"] == "1.2.3.4"
    assert instances[0]["Name"] == "aws-bootstrap-g4dn.xlarge"


def test_find_tagged_instances_empty():
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {"Reservations": []}
    assert find_tagged_instances(ec2, "aws-bootstrap-g4dn") == []


def test_terminate_tagged_instances():
    ec2 = MagicMock()
    ec2.terminate_instances.return_value = {
        "TerminatingInstances": [
            {
                "InstanceId": "i-abc123",
                "PreviousState": {"Name": "running"},
                "CurrentState": {"Name": "shutting-down"},
            }
        ]
    }
    changes = terminate_tagged_instances(ec2, ["i-abc123"])
    assert len(changes) == 1
    assert changes[0]["InstanceId"] == "i-abc123"
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-abc123"])
