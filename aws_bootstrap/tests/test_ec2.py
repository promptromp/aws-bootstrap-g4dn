"""Tests for EC2 helper functions."""

from __future__ import annotations
import io
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import botocore.exceptions
import click
import pytest

from aws_bootstrap.config import LaunchConfig
from aws_bootstrap.ec2 import (
    CapacityError,
    CLIError,
    RegionContext,
    RegionLaunch,
    find_tagged_instances,
    get_latest_ami,
    get_spot_price,
    launch_instance,
    launch_with_retry,
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


def test_launch_instance_quota_error_includes_quota_hint():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("MaxSpotInstanceCountExceeded")
    config = LaunchConfig(spot=True)
    with pytest.raises(click.ClickException, match="aws-bootstrap quota show"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_launch_instance_spot_quota_hint_has_type_spot():
    """Spot quota error hint suggests --type spot."""
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("MaxSpotInstanceCountExceeded")
    config = LaunchConfig(spot=True, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException, match="--type spot"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_launch_instance_on_demand_quota_hint_has_type_on_demand():
    """On-demand quota error hint suggests --type on-demand."""
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("VcpuLimitExceeded")
    config = LaunchConfig(spot=False, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException, match="--type on-demand"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_launch_instance_quota_hint_includes_family():
    """Quota error hint includes --family matching the instance type."""
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("MaxSpotInstanceCountExceeded")
    config = LaunchConfig(spot=True, instance_type="p5.48xlarge")
    with pytest.raises(click.ClickException, match="--family p"):
        launch_instance(ec2, config, "ami-test", "sg-test")


@patch("aws_bootstrap.ec2.is_text", return_value=False)
def test_launch_instance_on_demand_retry_quota_hint_type(_mock_is_text):
    """On-demand retry quota error hints --type on-demand, not spot."""
    ec2 = MagicMock()
    ec2.run_instances.side_effect = [
        _make_client_error("InsufficientInstanceCapacity", "No spot capacity"),
        _make_client_error("VcpuLimitExceeded"),
    ]
    config = LaunchConfig(spot=True, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException, match="--type on-demand"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def test_launch_instance_insufficient_capacity_on_demand():
    """On-demand launch with InsufficientInstanceCapacity gives a friendly error."""
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("InsufficientInstanceCapacity")
    config = LaunchConfig(spot=False, instance_type="p5.4xlarge")
    with pytest.raises(click.ClickException, match="Insufficient capacity for p5.4xlarge"):
        launch_instance(ec2, config, "ami-test", "sg-test")


@patch("aws_bootstrap.ec2.is_text", return_value=False)
def test_launch_instance_insufficient_capacity_on_demand_retry(_mock_is_text):
    """Spot fails, on-demand retry also gets InsufficientInstanceCapacity."""
    ec2 = MagicMock()
    ec2.run_instances.side_effect = [
        _make_client_error("InsufficientInstanceCapacity", "No spot capacity"),
        _make_client_error("InsufficientInstanceCapacity", "No on-demand capacity"),
    ]
    config = LaunchConfig(spot=True, instance_type="p5.4xlarge")
    with pytest.raises(click.ClickException, match="Neither spot nor on-demand"):
        launch_instance(ec2, config, "ami-test", "sg-test")


def _region_ctx(region: str, run_side_effect):
    """RegionContext whose ec2 client's run_instances has the given side effect."""
    client = MagicMock()
    client.run_instances.side_effect = run_side_effect
    return RegionContext(region=region, ec2_client=client, ami={"ImageId": f"ami-{region}"}, sg_id="sg-1")


def _ok_instance(instance_id="i-ok"):
    return {"Instances": [{"InstanceId": instance_id}]}


def test_launch_with_retry_falls_through_to_second_region_on_capacity():
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity")),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance("i-east")]),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True)
    calls: list[str] = []

    result = launch_with_retry(
        config,
        lambda r: contexts[r],
        on_attempt=lambda region, market, attempt: calls.append(f"{region}:{market}"),
    )
    assert isinstance(result, RegionLaunch)
    assert result.region == "us-east-1"
    assert result.instance["InstanceId"] == "i-east"
    assert result.pricing == "spot"
    assert calls == ["us-west-2:spot", "us-east-1:spot"]


def test_launch_with_retry_quota_error_fails_fast_no_retry():
    """Quota errors are never retried, even across regions."""
    ctx = _region_ctx("us-west-2", _make_client_error("VcpuLimitExceeded"))
    second = _region_ctx("us-east-1", [_ok_instance()])
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=False)
    with pytest.raises(click.ClickException, match="vCPU quota exceeded"):
        launch_with_retry(config, lambda r: {"us-west-2": ctx, "us-east-1": second}[r])
    second.ec2_client.run_instances.assert_not_called()


def test_launch_with_retry_spot_max_price_too_low_fails_fast():
    ctx = _region_ctx("us-west-2", _make_client_error("SpotMaxPriceTooLow"))
    config = LaunchConfig(regions=("us-west-2",), spot=True)
    with pytest.raises(click.ClickException, match="exceeds the default maximum"):
        launch_with_retry(config, lambda r: ctx)


def test_launch_with_retry_wait_times_out_hard_fail():
    ctx = _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity"))
    config = LaunchConfig(regions=("us-west-2",), spot=True, wait=True, wait_timeout=100)
    ticks = iter([0.0, 0.0, 150.0])  # start, check#1 (<100 -> one wait), check#2 (>=100 -> timeout)
    slept: list[float] = []
    waits: list[int] = []

    with pytest.raises(click.ClickException, match="within 100s"):
        launch_with_retry(
            config,
            lambda r: ctx,
            on_wait=lambda cycle, s, e: waits.append(cycle),
            sleeper=slept.append,
            clock=lambda: next(ticks),
            rng=__import__("random").Random(0),
        )
    assert slept, "expected at least one backoff sleep before timeout"
    assert waits == [1]


def test_launch_with_retry_no_wait_on_demand_fallback_across_regions():
    contexts = {
        # [spot attempt, on-demand attempt]
        "us-west-2": _region_ctx(
            "us-west-2",
            [
                _make_client_error("InsufficientInstanceCapacity"),
                _make_client_error("InsufficientInstanceCapacity"),
            ],
        ),
        "us-east-1": _region_ctx(
            "us-east-1",
            [_make_client_error("InsufficientInstanceCapacity"), _ok_instance("i-od")],
        ),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True)

    result = launch_with_retry(
        config,
        lambda r: contexts[r],
        confirm_on_demand=lambda: True,
    )
    assert result.region == "us-east-1"
    assert result.pricing == "on-demand"


def test_launch_with_retry_declined_on_demand_cancels():
    ctx = _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity"))
    config = LaunchConfig(regions=("us-west-2",), spot=True)
    with pytest.raises(click.ClickException, match="Launch cancelled"):
        launch_with_retry(config, lambda r: ctx, confirm_on_demand=lambda: False)


def test_launch_with_retry_prepares_each_region_once():
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity")),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance()]),
    }
    prep_calls: list[str] = []

    def prepare(region):
        prep_calls.append(region)
        return contexts[region]

    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True, wait=True, wait_timeout=600)
    result = launch_with_retry(config, prepare, clock=lambda: 0.0)
    assert result.region == "us-east-1"
    assert sorted(prep_calls) == ["us-east-1", "us-west-2"]
    assert len(prep_calls) == 2  # each region prepared exactly once


def test_launch_with_retry_quota_hint_pins_failed_region():
    """Quota error suggestions include --region for the region that failed."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity")),
        "us-east-1": _region_ctx("us-east-1", _make_client_error("MaxSpotInstanceCountExceeded")),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: contexts[r])
    msg = exc.value.format_message()
    assert "in us-east-1" in msg
    assert "aws-bootstrap quota show --family gvt --region us-east-1" in msg
    assert "--type spot --desired-value 4 --region us-east-1" in msg


def test_quota_hint_without_region_omits_flag():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _make_client_error("MaxSpotInstanceCountExceeded")
    config = LaunchConfig(spot=True, instance_type="g4dn.xlarge", regions=("us-west-2",))
    with pytest.raises(click.ClickException) as exc:
        launch_instance(ec2, config, "ami-test", "sg-test")
    # Single-region back-compat path still pins the region (config.region).
    assert "--region us-west-2" in exc.value.format_message()


def test_launch_with_retry_capacity_error_carries_region_and_market():
    err = CapacityError("eu-west-1", "spot", "no capacity")
    assert err.region == "eu-west-1"
    assert err.market == "spot"


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
