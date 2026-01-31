"""Tests for EC2 helper functions."""

from __future__ import annotations
import io
from datetime import UTC, datetime
from unittest.mock import MagicMock

import botocore.exceptions
import click
import pytest

from aws_bootstrap.config import LaunchConfig
from aws_bootstrap.ec2 import (
    CLIError,
    find_tagged_instances,
    get_latest_ami,
    get_spot_price,
    launch_instance,
    list_amis,
    list_instance_types,
    terminate_tagged_instances,
)


def test_cli_error_is_click_exception():
    err = CLIError("something went wrong")
    assert isinstance(err, click.ClickException)
    assert err.format_message() == "something went wrong"


def test_cli_error_show_outputs_red():
    err = CLIError("bad input")
    buf = io.StringIO()
    err.show(file=buf)
    output = buf.getvalue()
    assert "Error: bad input" in output


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
                        "InstanceLifecycle": "spot",
                        "Placement": {"AvailabilityZone": "us-west-2a"},
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
    assert instances[0]["Lifecycle"] == "spot"
    assert instances[0]["AvailabilityZone"] == "us-west-2a"


def test_find_tagged_instances_on_demand_lifecycle():
    """On-demand instances have no InstanceLifecycle key; should default to 'on-demand'."""
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-ondemand",
                        "State": {"Name": "running"},
                        "InstanceType": "g4dn.xlarge",
                        "PublicIpAddress": "5.6.7.8",
                        "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
                        "Placement": {"AvailabilityZone": "us-west-2b"},
                        "Tags": [
                            {"Key": "Name", "Value": "aws-bootstrap-g4dn.xlarge"},
                        ],
                    }
                ]
            }
        ]
    }
    instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    assert len(instances) == 1
    assert instances[0]["Lifecycle"] == "on-demand"
    assert instances[0]["AvailabilityZone"] == "us-west-2b"


def test_find_tagged_instances_empty():
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {"Reservations": []}
    assert find_tagged_instances(ec2, "aws-bootstrap-g4dn") == []


def test_get_spot_price_returns_price():
    ec2 = MagicMock()
    ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": [{"SpotPrice": "0.1578"}]}
    price = get_spot_price(ec2, "g4dn.xlarge", "us-west-2a")
    assert price == 0.1578
    ec2.describe_spot_price_history.assert_called_once()


def test_get_spot_price_returns_none_when_empty():
    ec2 = MagicMock()
    ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": []}
    price = get_spot_price(ec2, "g4dn.xlarge", "us-west-2a")
    assert price is None


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


# ---------------------------------------------------------------------------
# list_instance_types
# ---------------------------------------------------------------------------


def test_list_instance_types_returns_sorted():
    ec2 = MagicMock()
    paginator = MagicMock()
    ec2.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {
            "InstanceTypes": [
                {
                    "InstanceType": "g4dn.xlarge",
                    "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384},
                    "GpuInfo": {"Gpus": [{"Count": 1, "Name": "T4", "MemoryInfo": {"SizeInMiB": 16384}}]},
                },
                {
                    "InstanceType": "g4dn.2xlarge",
                    "VCpuInfo": {"DefaultVCpus": 8},
                    "MemoryInfo": {"SizeInMiB": 32768},
                    "GpuInfo": {"Gpus": [{"Count": 1, "Name": "T4", "MemoryInfo": {"SizeInMiB": 16384}}]},
                },
            ]
        }
    ]
    results = list_instance_types(ec2, "g4dn")
    assert len(results) == 2
    # sorted by name — 2xlarge < xlarge lexicographically
    assert results[0]["InstanceType"] == "g4dn.2xlarge"
    assert results[1]["InstanceType"] == "g4dn.xlarge"
    assert results[1]["VCpuCount"] == 4
    assert results[1]["MemoryMiB"] == 16384
    assert "T4" in results[1]["GpuSummary"]


def test_list_instance_types_no_gpu():
    ec2 = MagicMock()
    paginator = MagicMock()
    ec2.get_paginator.return_value = paginator
    paginator.paginate.return_value = [
        {
            "InstanceTypes": [
                {
                    "InstanceType": "t3.medium",
                    "VCpuInfo": {"DefaultVCpus": 2},
                    "MemoryInfo": {"SizeInMiB": 4096},
                },
            ]
        }
    ]
    results = list_instance_types(ec2, "t3")
    assert len(results) == 1
    assert results[0]["GpuSummary"] == ""


def test_list_instance_types_empty():
    ec2 = MagicMock()
    paginator = MagicMock()
    ec2.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{"InstanceTypes": []}]
    results = list_instance_types(ec2, "nonexistent")
    assert results == []


# ---------------------------------------------------------------------------
# list_amis
# ---------------------------------------------------------------------------


def test_list_amis_sorted_newest_first():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {
        "Images": [
            {
                "ImageId": "ami-old",
                "Name": "DL AMI old",
                "CreationDate": "2024-01-01T00:00:00Z",
                "Architecture": "x86_64",
            },
            {
                "ImageId": "ami-new",
                "Name": "DL AMI new",
                "CreationDate": "2025-06-01T00:00:00Z",
                "Architecture": "x86_64",
            },
        ]
    }
    results = list_amis(ec2, "DL AMI*")
    assert len(results) == 2
    assert results[0]["ImageId"] == "ami-new"
    assert results[1]["ImageId"] == "ami-old"


def test_list_amis_empty():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {"Images": []}
    results = list_amis(ec2, "nonexistent*")
    assert results == []


def test_list_amis_limited_to_20():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {
        "Images": [
            {
                "ImageId": f"ami-{i:03d}",
                "Name": f"AMI {i}",
                "CreationDate": f"2025-01-{i + 1:02d}T00:00:00Z",
                "Architecture": "x86_64",
            }
            for i in range(25)
        ]
    }
    results = list_amis(ec2, "AMI*")
    assert len(results) == 20


def test_list_amis_uses_owner_hint_for_deep_learning():
    ec2 = MagicMock()
    ec2.describe_images.return_value = {"Images": []}
    list_amis(ec2, "Deep Learning Base*")
    call_kwargs = ec2.describe_images.call_args[1]
    assert call_kwargs["Owners"] == ["amazon"]
