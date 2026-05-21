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
    _run_instances,
    delete_cluster_placement_group,
    ensure_cluster_placement_group,
    ensure_cluster_security_group_rule,
    find_tagged_instances,
    find_tagged_instances_in_regions,
    get_latest_ami,
    get_spot_price,
    launch_instance,
    launch_with_retry,
    list_amis,
    list_enabled_regions,
    list_instance_types,
    resolve_ebs_placement_az,
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
    return RegionContext(
        region=region, ec2_client=client, ami={"ImageId": f"ami-{region}"}, sg_id="sg-1", key_name="aws-bootstrap-key"
    )


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


def test_launch_with_retry_quota_skips_to_next_region():
    """A per-region quota error moves on to the next region (no longer fail-fast)."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("VcpuLimitExceeded")),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance("i-east")]),
    }
    fatal: list[tuple[str, str]] = []
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=False)
    result = launch_with_retry(
        config,
        lambda r: contexts[r],
        on_region_fatal=lambda region, kind, msg: fatal.append((region, kind)),
    )
    assert result.region == "us-east-1"
    contexts["us-east-1"].ec2_client.run_instances.assert_called_once()
    assert fatal == [("us-west-2", "quota")]


def test_launch_with_retry_all_regions_quota_aggregated_hard_fail():
    """Every region quota-blocked -> hard fail listing each region + pinned hints."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("VcpuLimitExceeded")),
        "us-east-1": _region_ctx("us-east-1", _make_client_error("VcpuLimitExceeded")),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=False, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: contexts[r])
    msg = exc.value.format_message()
    assert "us-west-2: on-demand quota exceeded" in msg
    assert "us-east-1: on-demand quota exceeded" in msg
    assert "aws-bootstrap quota show --family gvt --region us-west-2" in msg
    assert "aws-bootstrap quota show --family gvt --region us-east-1" in msg


def test_launch_with_retry_spot_price_skips_to_next_region():
    """SpotMaxPriceTooLow is region-fatal but the next region is still tried."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("SpotMaxPriceTooLow")),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance("i-east")]),
    }
    fatal: list[tuple[str, str]] = []
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True)
    result = launch_with_retry(
        config,
        lambda r: contexts[r],
        on_region_fatal=lambda region, kind, msg: fatal.append((region, kind)),
    )
    assert result.region == "us-east-1"
    assert result.pricing == "spot"
    assert fatal == [("us-west-2", "price")]


def test_launch_with_retry_single_region_price_emits_hint_then_fails():
    """Price-fatal single region: hint surfaced via callback, then hard fail."""
    ctx = _region_ctx("us-west-2", _make_client_error("SpotMaxPriceTooLow"))
    config = LaunchConfig(regions=("us-west-2",), spot=True)
    fatal: list[tuple[str, str, str]] = []
    with pytest.raises(click.ClickException, match="Launch cancelled"):
        launch_with_retry(
            config,
            lambda r: ctx,
            on_region_fatal=lambda region, kind, msg: fatal.append((region, kind, msg)),
            confirm_on_demand=lambda: False,
        )
    assert fatal[0][0] == "us-west-2"
    assert fatal[0][1] == "price"
    assert "exceeds the default maximum" in fatal[0][2]


def test_launch_with_retry_on_demand_quota_aggregated_after_spot_capacity():
    """Spot capacity miss -> on-demand fallback hits quota -> aggregated hint."""
    contexts = {
        "us-west-2": _region_ctx(
            "us-west-2",
            [_make_client_error("InsufficientInstanceCapacity"), _make_client_error("VcpuLimitExceeded")],
        ),
    }
    config = LaunchConfig(regions=("us-west-2",), spot=True, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: contexts[r], confirm_on_demand=lambda: True)
    msg = exc.value.format_message()
    assert "on-demand quota exceeded" in msg
    assert "aws-bootstrap quota show --family gvt --region us-west-2" in msg


def test_launch_with_retry_wait_only_retries_capacity_regions():
    """With --wait, quota-blocked regions are dropped; only capacity-limited
    regions are re-swept, and on_wait reports retried vs skipped accurately
    (regression: heartbeat used to list every region as if still swept)."""
    contexts = {
        "us-east-1": _region_ctx("us-east-1", _make_client_error("MaxSpotInstanceCountExceeded")),
        "us-west-1": _region_ctx("us-west-1", _make_client_error("MaxSpotInstanceCountExceeded")),
        "us-west-2": _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity")),
    }
    config = LaunchConfig(regions=("us-east-1", "us-west-1", "us-west-2"), spot=True, wait=True, wait_timeout=100)
    ticks = iter([0.0, 0.0, 150.0])
    waits: list[tuple] = []
    with pytest.raises(click.ClickException, match="within 100s"):
        launch_with_retry(
            config,
            lambda r: contexts[r],
            on_wait=lambda c, s, e, retrying, skipped: waits.append((tuple(retrying), tuple(skipped))),
            sleeper=lambda _s: None,
            clock=lambda: next(ticks),
            rng=__import__("random").Random(0),
        )
    assert waits == [(("us-west-2",), ("us-east-1", "us-west-1"))]
    # Quota-fatal regions attempted once only (not re-swept each cycle).
    contexts["us-east-1"].ec2_client.run_instances.assert_called_once()
    contexts["us-west-1"].ec2_client.run_instances.assert_called_once()
    assert contexts["us-west-2"].ec2_client.run_instances.call_count >= 2


def test_launch_with_retry_wait_times_out_hard_fail():
    ctx = _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity"))
    config = LaunchConfig(regions=("us-west-2",), spot=True, wait=True, wait_timeout=100)
    ticks = iter([0.0, 0.0, 150.0])  # start, check#1 (<100 -> one wait), check#2 (>=100 -> timeout)
    slept: list[float] = []
    waits: list[tuple] = []

    with pytest.raises(click.ClickException, match="within 100s"):
        launch_with_retry(
            config,
            lambda r: ctx,
            on_wait=lambda cycle, s, e, retrying, skipped: waits.append((cycle, tuple(retrying), tuple(skipped))),
            sleeper=slept.append,
            clock=lambda: next(ticks),
            rng=__import__("random").Random(0),
        )
    assert slept, "expected at least one backoff sleep before timeout"
    # One wait cycle; only the capacity-limited region is retried, none skipped.
    assert waits == [(1, ("us-west-2",), ())]


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
    """Aggregated failure pins the quota hint to the region that failed."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", _make_client_error("InsufficientInstanceCapacity")),
        "us-east-1": _region_ctx("us-east-1", _make_client_error("VcpuLimitExceeded")),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=False, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: contexts[r])
    msg = exc.value.format_message()
    assert "us-east-1: on-demand quota exceeded" in msg
    assert "aws-bootstrap quota show --family gvt --region us-east-1" in msg
    assert "--type on-demand --desired-value <N> --region us-east-1" in msg


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


# ---------------------------------------------------------------------------
# list_enabled_regions
# ---------------------------------------------------------------------------


def test_list_enabled_regions_sorted():
    ec2 = MagicMock()
    ec2.describe_regions.return_value = {
        "Regions": [
            {"RegionName": "us-west-2"},
            {"RegionName": "eu-west-1"},
            {"RegionName": "us-east-1"},
        ]
    }
    regions = list_enabled_regions(ec2)
    assert regions == ["eu-west-1", "us-east-1", "us-west-2"]
    ec2.describe_regions.assert_called_once_with(AllRegions=False)


def test_list_enabled_regions_empty():
    ec2 = MagicMock()
    ec2.describe_regions.return_value = {"Regions": []}
    assert list_enabled_regions(ec2) == []


# ---------------------------------------------------------------------------
# find_tagged_instances_in_regions
# ---------------------------------------------------------------------------


def _instance_response(instance_id: str, az: str):
    return {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": instance_id,
                        "State": {"Name": "running"},
                        "InstanceType": "g4dn.xlarge",
                        "PublicIpAddress": "1.2.3.4",
                        "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
                        "InstanceLifecycle": "spot",
                        "Placement": {"AvailabilityZone": az},
                        "Tags": [{"Key": "created-by", "Value": "aws-bootstrap-g4dn"}],
                    }
                ]
            }
        ]
    }


def _session_with_region_clients(clients_by_region: dict[str, MagicMock]) -> MagicMock:
    session = MagicMock()

    def client_factory(_service, region_name=None, **_kwargs):
        return clients_by_region[region_name]

    session.client.side_effect = client_factory
    return session


def test_find_tagged_instances_in_regions_aggregates_and_labels():
    east = MagicMock()
    east.describe_instances.return_value = _instance_response("i-east", "us-east-1a")
    west = MagicMock()
    west.describe_instances.return_value = _instance_response("i-west", "us-west-2a")
    session = _session_with_region_clients({"us-east-1": east, "us-west-2": west})

    instances, failures = find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", ["us-east-1", "us-west-2"])

    assert failures == []
    assert {i["InstanceId"] for i in instances} == {"i-east", "i-west"}
    by_id = {i["InstanceId"]: i for i in instances}
    assert by_id["i-east"]["Region"] == "us-east-1"
    assert by_id["i-west"]["Region"] == "us-west-2"
    # Sorted by region then launch time
    assert [i["Region"] for i in instances] == ["us-east-1", "us-west-2"]


def test_find_tagged_instances_in_regions_captures_failure():
    ok = MagicMock()
    ok.describe_instances.return_value = _instance_response("i-ok", "us-east-1a")
    bad = MagicMock()
    bad.describe_instances.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "AuthFailure", "Message": "not authorized"}},
        "DescribeInstances",
    )
    session = _session_with_region_clients({"us-east-1": ok, "ap-south-1": bad})

    instances, failures = find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", ["us-east-1", "ap-south-1"])

    assert [i["InstanceId"] for i in instances] == ["i-ok"]
    assert len(failures) == 1
    assert failures[0]["region"] == "ap-south-1"
    assert "AuthFailure" in failures[0]["error"]


def test_find_tagged_instances_in_regions_empty_region_list():
    session = MagicMock()
    instances, failures = find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", [])
    assert instances == []
    assert failures == []
    session.client.assert_not_called()


def test_find_tagged_instances_in_regions_propagates_credential_error():
    """Non-ClientError exceptions (e.g. missing credentials) are not swallowed."""
    bad = MagicMock()
    bad.describe_instances.side_effect = botocore.exceptions.NoCredentialsError()
    session = _session_with_region_clients({"us-east-1": bad})

    with pytest.raises(botocore.exceptions.NoCredentialsError):
        find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", ["us-east-1"])


# ---------------------------------------------------------------------------
# EBS AZ pinning: an existing data volume is AZ-scoped, so the instance must be
# launched in the volume's AZ. resolve_ebs_placement_az looks up the AZ once;
# _run_instances pins the run_instances request to it via Placement.
# ---------------------------------------------------------------------------


def test_resolve_ebs_placement_az_returns_volume_az():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {"Volumes": [{"VolumeId": "vol-abc", "AvailabilityZone": "us-east-1c"}]}
    assert resolve_ebs_placement_az(ec2, "vol-abc", "us-east-1") == "us-east-1c"
    ec2.describe_volumes.assert_called_once_with(VolumeIds=["vol-abc"])


def test_resolve_ebs_placement_az_not_found_names_region():
    ec2 = MagicMock()
    ec2.describe_volumes.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "InvalidVolume.NotFound", "Message": "not found"}},
        "DescribeVolumes",
    )
    with pytest.raises(CLIError, match="EBS volume not found: vol-x in us-west-2"):
        resolve_ebs_placement_az(ec2, "vol-x", "us-west-2")


def test_resolve_ebs_placement_az_empty_response():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {"Volumes": []}
    with pytest.raises(CLIError, match="EBS volume not found"):
        resolve_ebs_placement_az(ec2, "vol-x", "us-west-2")


def test_run_instances_pins_placement_az_when_provided():
    ec2 = MagicMock()
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-test"}]}
    config = LaunchConfig(ebs_volume_id="vol-abc", spot=True)

    _run_instances(ec2, config, "ami-test", "sg-test", "us-east-1", True, "key-test", "us-east-1c")

    # _run_instances no longer resolves the AZ itself — it is passed in.
    ec2.describe_volumes.assert_not_called()
    assert ec2.run_instances.call_args[1]["Placement"] == {"AvailabilityZone": "us-east-1c"}


def test_run_instances_omits_placement_without_az():
    ec2 = MagicMock()
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-test"}]}
    config = LaunchConfig(spot=True)

    _run_instances(ec2, config, "ami-test", "sg-test", "us-east-1", True, "key-test")

    assert "Placement" not in ec2.run_instances.call_args[1]


def test_launch_with_retry_passes_placement_az_to_run_instances():
    """The AZ resolved into RegionContext is threaded through to run_instances."""
    client = MagicMock()
    client.run_instances.return_value = _ok_instance("i-east")
    ctx = RegionContext(
        region="us-east-1",
        ec2_client=client,
        ami={"ImageId": "ami-east"},
        sg_id="sg-1",
        key_name="aws-bootstrap-key",
        placement_az="us-east-1c",
    )
    config = LaunchConfig(regions=("us-east-1",), spot=True, ebs_volume_id="vol-abc")

    result = launch_with_retry(config, lambda r: ctx)

    assert result.region == "us-east-1"
    assert client.run_instances.call_args[1]["Placement"] == {"AvailabilityZone": "us-east-1c"}


# ---------------------------------------------------------------------------
# Cluster placement-group lifecycle
# ---------------------------------------------------------------------------


def test_ensure_cluster_placement_group_creates_when_absent():
    ec2 = MagicMock()
    ec2.describe_placement_groups.return_value = {"PlacementGroups": []}
    name = ensure_cluster_placement_group(ec2, "aws-bootstrap-cluster-ml1", "aws-bootstrap-g4dn")
    assert name == "aws-bootstrap-cluster-ml1"
    create_kwargs = ec2.create_placement_group.call_args[1]
    assert create_kwargs["GroupName"] == "aws-bootstrap-cluster-ml1"
    assert create_kwargs["Strategy"] == "cluster"


def test_ensure_cluster_placement_group_reuses_existing():
    ec2 = MagicMock()
    ec2.describe_placement_groups.return_value = {
        "PlacementGroups": [{"GroupName": "aws-bootstrap-cluster-ml1", "State": "available"}]
    }
    name = ensure_cluster_placement_group(ec2, "aws-bootstrap-cluster-ml1", "aws-bootstrap-g4dn")
    assert name == "aws-bootstrap-cluster-ml1"
    ec2.create_placement_group.assert_not_called()


def test_delete_cluster_placement_group_ignores_unknown():
    ec2 = MagicMock()
    ec2.delete_placement_group.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "InvalidPlacementGroup.Unknown", "Message": "nope"}},
        "DeletePlacementGroup",
    )
    # Must not raise.
    delete_cluster_placement_group(ec2, "aws-bootstrap-cluster-ml1")
    ec2.delete_placement_group.assert_called_once_with(GroupName="aws-bootstrap-cluster-ml1")


# ---------------------------------------------------------------------------
# Intra-cluster security-group rule
# ---------------------------------------------------------------------------


def test_ensure_cluster_sg_rule_authorizes_self_reference():
    ec2 = MagicMock()
    ensure_cluster_security_group_rule(ec2, "sg-123")
    kwargs = ec2.authorize_security_group_ingress.call_args[1]
    assert kwargs["GroupId"] == "sg-123"
    perm = kwargs["IpPermissions"][0]
    assert perm["IpProtocol"] == "-1"
    assert perm["UserIdGroupPairs"][0]["GroupId"] == "sg-123"


def test_ensure_cluster_sg_rule_idempotent_on_duplicate():
    ec2 = MagicMock()
    ec2.authorize_security_group_ingress.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "InvalidPermission.Duplicate", "Message": "exists"}},
        "AuthorizeSecurityGroupIngress",
    )
    # Must not raise.
    ensure_cluster_security_group_rule(ec2, "sg-123")
