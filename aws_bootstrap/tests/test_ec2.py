"""Tests for EC2 helper functions."""

from __future__ import annotations
import fnmatch
import io
import random
from datetime import UTC, datetime
from unittest.mock import MagicMock

import botocore.exceptions
import click
import pytest

from aws_bootstrap.config import DEFAULT_AMI_FILTER, LaunchConfig
from aws_bootstrap.constants import TAG_CREATED_BY, TAG_VALUE
from aws_bootstrap.ec2 import (
    CapacityError,
    CLIError,
    RegionContext,
    RegionFatalError,
    RegionLaunch,
    _gpu_summary,
    _quota_error_message,
    _quota_hint,
    _run_instances,
    delete_cluster_placement_group,
    ensure_cluster_placement_group,
    ensure_cluster_security_group_rule,
    ensure_security_group,
    find_cluster_instances,
    find_tagged_instances,
    find_tagged_instances_in_regions,
    find_unowned_instances,
    get_latest_ami,
    get_spot_price,
    instance_type_architecture,
    launch_with_retry,
    list_amis,
    list_clusters,
    list_enabled_regions,
    list_instance_types,
    terminate_tagged_instances,
)
from aws_bootstrap.iam import InstanceProfileError
from aws_bootstrap.tests.helpers import client_error, session_with_clients


@pytest.fixture
def ec2():
    """A bare mocked EC2 client."""
    return MagicMock()


def test_cli_error_is_a_click_error_shown_with_an_error_prefix():
    err = CLIError("bad input")
    buf = io.StringIO()
    err.show(file=buf)
    assert isinstance(err, click.ClickException) and err.exit_code == 1
    assert buf.getvalue() == "Error: bad input\n"


# ---------------------------------------------------------------------------
# AMI lookup
# ---------------------------------------------------------------------------


def test_get_latest_ami_picks_newest(ec2):
    ec2.describe_images.return_value = {
        "Images": [
            {"ImageId": "ami-old", "Name": "DL AMI old", "CreationDate": "2024-01-01T00:00:00Z"},
            {"ImageId": "ami-new", "Name": "DL AMI new", "CreationDate": "2025-06-01T00:00:00Z"},
            {"ImageId": "ami-mid", "Name": "DL AMI mid", "CreationDate": "2025-01-01T00:00:00Z"},
        ]
    }
    assert get_latest_ami(ec2, "DL AMI*")["ImageId"] == "ami-new"


def test_get_latest_ami_no_results(ec2):
    ec2.describe_images.return_value = {"Images": []}
    with pytest.raises(click.ClickException, match="No x86_64 AMI found"):
        get_latest_ami(ec2, "nonexistent*")


# ---------------------------------------------------------------------------
# Instance-type architecture / architecture-aware AMI selection
#
# An AMI must match its instance type's CPU architecture. get_latest_ami used to
# hardcode x86_64, so an arm64 GPU type (g5g / Grace-based P-series) silently got
# an x86_64 AMI and failed minutes later inside run_instances.
# ---------------------------------------------------------------------------


def _instance_type_response(arches: list[str]) -> dict:
    return {"InstanceTypes": [{"ProcessorInfo": {"SupportedArchitectures": arches}}]}


@pytest.mark.parametrize(
    "response,expected",
    [
        pytest.param(_instance_type_response(["x86_64"]), "x86_64", id="x86_64"),
        pytest.param(_instance_type_response(["arm64"]), "arm64", id="arm64"),
        # Mixed-arch types keep x86_64: far more AMIs are published for it.
        pytest.param(_instance_type_response(["arm64", "x86_64"]), "x86_64", id="mixed"),
        pytest.param(_instance_type_response([]), "x86_64", id="no-architectures"),
        pytest.param({"InstanceTypes": []}, "x86_64", id="empty-response"),
        # An unknown type in this region must not abort the launch sweep.
        pytest.param(client_error("InvalidInstanceType"), "x86_64", id="unknown-type"),
    ],
)
def test_instance_type_architecture(ec2, response, expected):
    if isinstance(response, Exception):
        ec2.describe_instance_types.side_effect = response
    else:
        ec2.describe_instance_types.return_value = response
    assert instance_type_architecture(ec2, "g5g.xlarge") == expected


@pytest.mark.parametrize("arch", ["x86_64", "arm64"])
def test_get_latest_ami_filters_on_requested_architecture(ec2, arch):
    ec2.describe_images.return_value = {
        "Images": [{"ImageId": "ami-1", "Name": "n", "CreationDate": "2025-06-01T00:00:00Z"}]
    }
    get_latest_ami(ec2, "DL AMI*", architecture=arch)
    filters = ec2.describe_images.call_args.kwargs["Filters"]
    assert {"Name": "architecture", "Values": [arch]} in filters


def test_get_latest_ami_arch_miss_explains_the_mismatch(ec2):
    """A custom filter written for x86_64 finds nothing for arm64 — the error must say why."""
    ec2.meta.region_name = "us-west-2"
    ec2.describe_images.return_value = {"Images": []}
    with pytest.raises(click.ClickException) as exc:
        get_latest_ami(ec2, "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*", architecture="arm64")
    msg = exc.value.format_message()
    assert "No arm64 AMI found" in msg
    assert "ARM64" in msg
    assert "--ami-filter to use the default" in msg
    assert "aws-bootstrap list amis --arch arm64 --region us-west-2" in msg
    # The old hint pointed at the PyTorch 2.10-2.12 DLAMIs, which AWS no longer patches.
    assert "PyTorch" not in msg


def test_get_latest_ami_miss_with_default_filter_blames_the_region(ec2):
    """The default spans every architecture, so if it finds nothing the region lacks the image."""
    ec2.meta.region_name = "eu-south-2"
    ec2.describe_images.return_value = {"Images": []}
    with pytest.raises(click.ClickException) as exc:
        get_latest_ami(ec2, DEFAULT_AMI_FILTER, architecture="x86_64")
    msg = exc.value.format_message()
    assert "try another --region" in msg
    assert "--ami-filter" not in msg
    assert "aws-bootstrap list amis --arch x86_64 --region eu-south-2" in msg


def test_no_ami_message_without_a_known_region(ec2):
    # meta.region_name is a MagicMock, not a str: no --region flag
    ec2.describe_images.return_value = {"Images": []}
    with pytest.raises(click.ClickException) as exc:
        get_latest_ami(ec2, "X*", architecture="arm64")
    assert exc.value.format_message().endswith("aws-bootstrap list amis --arch arm64")


@pytest.mark.parametrize(
    "name,matches",
    [
        ("Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261120", True),
        ("Deep Learning ARM64 Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261120", True),
        ("Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04) 20261120", False),
        ("Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu Pro 24.04) 20261118", False),
        ("Deep Learning ARM64 AMI OSS Nvidia Driver GPU PyTorch 2.12 (Ubuntu 24.04) 20261120", False),
    ],
)
def test_default_ami_filter_spans_architectures_but_not_other_images(name, matches):
    """One default for every architecture: the arch filter, not the name, picks the build.

    Glob semantics match EC2's name filter (`*` = any run of characters).
    """
    assert fnmatch.fnmatchcase(name, DEFAULT_AMI_FILTER) is matches


# ---------------------------------------------------------------------------
# Quota error messages
#
# These assert the *message* contract (label, --type, --family, region-pinned
# commands) against the pure builder. They used to go through the since-removed
# `launch_instance`; `launch_with_retry` covers the multi-region orchestration.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,spot,label,quota_type",
    [
        ("MaxSpotInstanceCountExceeded", True, "Spot instance quota exceeded", "spot"),
        ("VcpuLimitExceeded", False, "On-demand vCPU quota exceeded", "on-demand"),
    ],
    ids=["spot", "on-demand"],
)
def test_quota_message_label_and_type(code, spot, label, quota_type):
    msg = _quota_error_message(code, LaunchConfig(spot=spot), "us-west-2")
    assert label in msg
    assert f"--type {quota_type}" in msg
    assert "aws-bootstrap quota show" in msg


@pytest.mark.parametrize(
    ("instance_type", "family"),
    [
        ("g4dn.xlarge", "gvt"),
        ("g5.2xlarge", "gvt"),
        ("p5.48xlarge", "p"),
        ("dl1.24xlarge", "dl"),
        # A non-GPU type has no quota family; the hint still has to name one.
        ("t3.medium", "gvt"),
    ],
)
def test_quota_message_family_matches_instance_type(instance_type, family):
    msg = _quota_error_message("VcpuLimitExceeded", LaunchConfig(instance_type=instance_type), "us-west-2")
    assert f"--family {family}" in msg


def test_quota_message_pins_the_failed_region():
    msg = _quota_error_message("MaxSpotInstanceCountExceeded", LaunchConfig(), "eu-west-1")
    assert "in eu-west-1" in msg
    assert "--region eu-west-1" in msg


def test_quota_error_message_falls_back_to_config_region():
    """`region=None` still pins the hint, via config.regions[0]."""
    config = LaunchConfig(spot=True, instance_type="g4dn.xlarge", regions=("us-west-2",))
    assert "--region us-west-2" in _quota_error_message("MaxSpotInstanceCountExceeded", config, None)


def test_quota_hint_without_region_omits_flag():
    """With no region to pin, the suggested commands carry no --region flag."""
    hint = _quota_hint("spot", "gvt", None)
    assert "--region" not in hint
    assert "aws-bootstrap quota show --family gvt" in hint


# ---------------------------------------------------------------------------
# _run_instances: error mapping and placement
# ---------------------------------------------------------------------------


def test_run_instances_maps_capacity_error(ec2):
    """InsufficientInstanceCapacity is retryable (CapacityError), not fatal."""
    ec2.run_instances.side_effect = client_error("InsufficientInstanceCapacity")
    with pytest.raises(CapacityError) as exc:
        _run_instances(ec2, LaunchConfig(spot=True), "ami-1", "sg-1", "us-west-2", True, "k")
    assert exc.value.region == "us-west-2"
    assert exc.value.market == "spot"


@pytest.mark.parametrize(
    ("code", "kind"),
    [("MaxSpotInstanceCountExceeded", "quota"), ("VcpuLimitExceeded", "quota"), ("SpotMaxPriceTooLow", "price")],
)
def test_run_instances_maps_region_fatal_error(ec2, code, kind):
    """Quota / price are never worth waiting on -> RegionFatalError with a kind."""
    ec2.run_instances.side_effect = client_error(code)
    with pytest.raises(RegionFatalError) as exc:
        _run_instances(ec2, LaunchConfig(spot=True), "ami-1", "sg-1", "us-west-2", True, "k")
    assert exc.value.kind == kind


def test_run_instances_reraises_unknown_client_error(ec2):
    """An unmapped API error must not be silently reclassified as capacity."""
    ec2.run_instances.side_effect = client_error("UnauthorizedOperation")
    with pytest.raises(botocore.exceptions.ClientError):
        _run_instances(ec2, LaunchConfig(spot=True), "ami-1", "sg-1", "us-west-2", True, "k")


@pytest.mark.parametrize(
    "placement_az,placement_group,expected",
    [
        pytest.param("us-east-1c", None, {"AvailabilityZone": "us-east-1c"}, id="az"),
        pytest.param(
            "us-east-1c",
            "aws-bootstrap-cluster-ml1",
            {"AvailabilityZone": "us-east-1c", "GroupName": "aws-bootstrap-cluster-ml1"},
            id="az-and-group",
        ),
        pytest.param(None, "aws-bootstrap-cluster-ml1", {"GroupName": "aws-bootstrap-cluster-ml1"}, id="group-only"),
    ],
)
def test_run_instances_placement(ec2, placement_az, placement_group, expected):
    """EBS AZ pinning and cluster placement groups reach run_instances via Placement."""
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-test"}]}
    config = LaunchConfig(ebs_volume_id="vol-abc", spot=True)
    _run_instances(
        ec2,
        config,
        "ami-test",
        "sg-test",
        "us-east-1",
        True,
        "key-test",
        placement_az=placement_az,
        placement_group=placement_group,
    )
    # _run_instances no longer resolves the AZ itself — it is passed in.
    ec2.describe_volumes.assert_not_called()
    assert ec2.run_instances.call_args[1]["Placement"] == expected


def test_run_instances_omits_placement_without_az_or_group(ec2):
    """No Placement key at all: an empty/None one is a parameter-validation error."""
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-test"}]}
    _run_instances(ec2, LaunchConfig(spot=True), "ami-test", "sg-test", "us-east-1", True, "key-test")
    assert "Placement" not in ec2.run_instances.call_args[1]


PROFILE_ARN = "arn:aws:iam::123456789012:instance-profile/team/app"
PROFILE_REJECTED = client_error(
    "InvalidParameterValue",
    "Value (app) for parameter iamInstanceProfile.name is invalid. Invalid IAM Instance Profile name",
)


@pytest.mark.parametrize("spot", [True, False], ids=["spot", "on-demand"])
@pytest.mark.parametrize(
    "profile,expected",
    [
        pytest.param(None, None, id="none"),
        pytest.param("app", {"Name": "app"}, id="name"),
        pytest.param(PROFILE_ARN, {"Arn": PROFILE_ARN}, id="arn"),
    ],
)
def test_run_instances_attaches_the_instance_profile(ec2, profile, expected, spot):
    ec2.run_instances.return_value = {"Instances": [{"InstanceId": "i-test"}]}
    _run_instances(ec2, LaunchConfig(iam_instance_profile=profile), "ami-1", "sg-1", "us-west-2", spot, "k")
    assert ec2.run_instances.call_args[1].get("IamInstanceProfile") == expected


@pytest.mark.parametrize(
    "error,match",
    [
        pytest.param(PROFILE_REJECTED, "EC2 rejected IAM instance profile 'app'", id="invalid-profile"),
        pytest.param(
            client_error("InvalidParameterValue", "Value (x) for parameter IamInstanceProfile.Arn is invalid"),
            "EC2 rejected IAM instance profile 'app'",
            id="invalid-profile-other-casing",
        ),
        pytest.param(
            client_error("UnauthorizedOperation", "Encoded authorization failure message: xyz"),
            "often a missing iam:PassRole",
            id="pass-role-denied",
        ),
    ],
)
def test_run_instances_profile_errors_become_global_errors(ec2, error, match):
    ec2.run_instances.side_effect = error
    with pytest.raises(InstanceProfileError, match=match):
        _run_instances(ec2, LaunchConfig(iam_instance_profile="app"), "ami-1", "sg-1", "us-west-2", True, "k")


def test_run_instances_does_not_blame_the_profile_for_other_bad_parameters(ec2):
    ec2.run_instances.side_effect = client_error(
        "InvalidParameterValue", "Value (x) for parameter blockDeviceMapping is invalid"
    )
    with pytest.raises(botocore.exceptions.ClientError):
        _run_instances(ec2, LaunchConfig(iam_instance_profile="app"), "ami-1", "sg-1", "us-west-2", True, "k")


# ---------------------------------------------------------------------------
# launch_with_retry
# ---------------------------------------------------------------------------


def _region_ctx(region: str, run_side_effect, **ctx_fields):
    """RegionContext whose ec2 client's run_instances has the given side effect."""
    client = MagicMock()
    client.run_instances.side_effect = run_side_effect
    return RegionContext(
        region=region,
        ec2_client=client,
        ami={"ImageId": f"ami-{region}"},
        sg_id="sg-1",
        key_name="aws-bootstrap-key",
        **ctx_fields,
    )


def _ok_instance(instance_id="i-ok"):
    return {"Instances": [{"InstanceId": instance_id}]}


def test_launch_with_retry_falls_through_to_second_region_on_capacity():
    contexts = {
        "us-west-2": _region_ctx("us-west-2", client_error("InsufficientInstanceCapacity")),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance("i-east")]),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True)
    calls: list[str] = []

    result = launch_with_retry(
        config,
        lambda r: contexts[r],
        on_attempt=lambda region, market, attempt, az: calls.append(f"{region}:{market}"),
    )
    assert isinstance(result, RegionLaunch)
    assert result.region == "us-east-1"
    assert result.instance["InstanceId"] == "i-east"
    assert result.pricing == "spot"
    assert calls == ["us-west-2:spot", "us-east-1:spot"]


@pytest.mark.parametrize(
    "code,spot,kind",
    [
        # A per-region quota error moves on to the next region (no longer fail-fast).
        pytest.param("VcpuLimitExceeded", False, "quota", id="quota"),
        # SpotMaxPriceTooLow is region-fatal but the next region is still tried.
        pytest.param("SpotMaxPriceTooLow", True, "price", id="spot-price"),
    ],
)
def test_launch_with_retry_region_fatal_skips_to_next_region(code, spot, kind):
    contexts = {
        "us-west-2": _region_ctx("us-west-2", client_error(code)),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance("i-east")]),
    }
    fatal: list[tuple[str, str]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1"), spot=spot),
        lambda r: contexts[r],
        on_region_fatal=lambda region, kind, msg: fatal.append((region, kind)),
    )
    assert result.region == "us-east-1"
    assert result.pricing == ("spot" if spot else "on-demand")
    contexts["us-east-1"].ec2_client.run_instances.assert_called_once()
    assert fatal == [("us-west-2", kind)]


def test_launch_with_retry_all_regions_quota_aggregated_hard_fail():
    """Every region quota-blocked -> hard fail listing each region + pinned hints."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", client_error("VcpuLimitExceeded")),
        "us-east-1": _region_ctx("us-east-1", client_error("VcpuLimitExceeded")),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=False, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: contexts[r])
    msg = exc.value.format_message()
    assert "us-west-2: on-demand quota exceeded" in msg
    assert "us-east-1: on-demand quota exceeded" in msg
    assert "aws-bootstrap quota show --family gvt --region us-west-2" in msg
    assert "aws-bootstrap quota show --family gvt --region us-east-1" in msg


def test_launch_with_retry_quota_hint_pins_failed_region():
    """Aggregated failure pins the quota hint to the region that failed."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", client_error("InsufficientInstanceCapacity")),
        "us-east-1": _region_ctx("us-east-1", client_error("VcpuLimitExceeded")),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=False, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: contexts[r])
    msg = exc.value.format_message()
    assert "us-east-1: on-demand quota exceeded" in msg
    assert "aws-bootstrap quota show --family gvt --region us-east-1" in msg
    assert "--type on-demand --desired-value <N> --region us-east-1" in msg


def test_launch_with_retry_single_region_price_emits_hint_then_fails():
    """Price-fatal single region: hint surfaced via callback, then hard fail."""
    ctx = _region_ctx("us-west-2", client_error("SpotMaxPriceTooLow"))
    fatal: list[tuple[str, str, str]] = []
    with pytest.raises(click.ClickException, match="Launch cancelled"):
        launch_with_retry(
            LaunchConfig(regions=("us-west-2",), spot=True),
            lambda r: ctx,
            on_region_fatal=lambda region, kind, msg: fatal.append((region, kind, msg)),
            confirm_on_demand=lambda: False,
        )
    region, kind, msg = fatal[0]
    assert (region, kind) == ("us-west-2", "price")
    assert "exceeds the default maximum" in msg


def test_launch_with_retry_on_demand_quota_aggregated_after_spot_capacity():
    """Spot capacity miss -> on-demand fallback hits quota -> aggregated hint."""
    ctx = _region_ctx("us-west-2", [client_error("InsufficientInstanceCapacity"), client_error("VcpuLimitExceeded")])
    config = LaunchConfig(regions=("us-west-2",), spot=True, instance_type="g4dn.xlarge")
    with pytest.raises(click.ClickException) as exc:
        launch_with_retry(config, lambda r: ctx, confirm_on_demand=lambda: True)
    msg = exc.value.format_message()
    assert "on-demand quota exceeded" in msg
    assert "aws-bootstrap quota show --family gvt --region us-west-2" in msg


def test_launch_with_retry_wait_only_retries_capacity_regions():
    """With --wait, quota-blocked regions are dropped; only capacity-limited
    regions are re-swept, and on_wait reports retried vs skipped accurately
    (regression: heartbeat used to list every region as if still swept)."""
    contexts = {
        "us-east-1": _region_ctx("us-east-1", client_error("MaxSpotInstanceCountExceeded")),
        "us-west-1": _region_ctx("us-west-1", client_error("MaxSpotInstanceCountExceeded")),
        "us-west-2": _region_ctx("us-west-2", client_error("InsufficientInstanceCapacity")),
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
            rng=random.Random(0),
        )
    assert waits == [(("us-west-2",), ("us-east-1", "us-west-1"))]
    # Quota-fatal regions attempted once only (not re-swept each cycle).
    contexts["us-east-1"].ec2_client.run_instances.assert_called_once()
    contexts["us-west-1"].ec2_client.run_instances.assert_called_once()
    assert contexts["us-west-2"].ec2_client.run_instances.call_count >= 2


def test_launch_with_retry_wait_times_out_hard_fail():
    ctx = _region_ctx("us-west-2", client_error("InsufficientInstanceCapacity"))
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
            rng=random.Random(0),
        )
    assert slept, "expected at least one backoff sleep before timeout"
    # One wait cycle; only the capacity-limited region is retried, none skipped.
    assert waits == [(1, ("us-west-2",), ())]


def test_launch_with_retry_no_wait_on_demand_fallback_across_regions():
    capacity = client_error("InsufficientInstanceCapacity")
    contexts = {
        # [spot attempt, on-demand attempt]
        "us-west-2": _region_ctx("us-west-2", [capacity, capacity]),
        "us-east-1": _region_ctx("us-east-1", [capacity, _ok_instance("i-od")]),
    }
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True)
    result = launch_with_retry(config, lambda r: contexts[r], confirm_on_demand=lambda: True)
    assert result.region == "us-east-1"
    assert result.pricing == "on-demand"


def test_launch_with_retry_declined_on_demand_cancels():
    ctx = _region_ctx("us-west-2", client_error("InsufficientInstanceCapacity"))
    config = LaunchConfig(regions=("us-west-2",), spot=True)
    with pytest.raises(click.ClickException, match="Launch cancelled"):
        launch_with_retry(config, lambda r: ctx, confirm_on_demand=lambda: False)


def test_launch_with_retry_prepares_each_region_once_across_wait_sweeps():
    capacity = client_error("InsufficientInstanceCapacity")
    contexts = {
        "us-west-2": _region_ctx("us-west-2", capacity),  # never has capacity
        "us-east-1": _region_ctx("us-east-1", [capacity, _ok_instance()]),  # frees up on the 2nd sweep
    }
    prep_calls: list[str] = []

    def prepare(region):
        prep_calls.append(region)
        return contexts[region]

    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True, wait=True, wait_timeout=600)
    result = launch_with_retry(config, prepare, sleeper=lambda _s: None, clock=lambda: 0.0, rng=random.Random(0))
    assert result.region == "us-east-1"
    assert contexts["us-east-1"].ec2_client.run_instances.call_count == 2  # two sweeps happened
    assert sorted(prep_calls) == ["us-east-1", "us-west-2"]  # ...but each region was prepared once


@pytest.mark.parametrize(
    "ctx_fields,expected",
    [
        pytest.param({"placement_azs": ("us-east-1c",)}, {"AvailabilityZone": "us-east-1c"}, id="hard-pinned-az"),
        pytest.param(
            {"placement_azs": ("us-east-1c",), "placement_group": "aws-bootstrap-cluster-ml1"},
            {"AvailabilityZone": "us-east-1c", "GroupName": "aws-bootstrap-cluster-ml1"},
            id="hard-pinned-az-and-placement-group",
        ),
    ],
)
def test_launch_with_retry_sends_the_pinned_az_to_run_instances(ctx_fields, expected):
    """A hard-pinned AZ (an EBS volume's, or a cluster's first node's) and the cluster
    placement group in RegionContext reach run_instances as Placement."""
    ctx = _region_ctx("us-east-1", [_ok_instance("i-east")], **ctx_fields)
    result = launch_with_retry(LaunchConfig(regions=("us-east-1",), spot=True, ebs_volume_id="vol-abc"), lambda r: ctx)
    assert result.region == "us-east-1"
    assert ctx.ec2_client.run_instances.call_args[1]["Placement"] == expected


@pytest.mark.parametrize("wait", [False, True], ids=["no-wait", "wait"])
def test_launch_with_retry_profile_error_never_falls_through(wait):
    """A bad profile fails the same in every region and market: stop at once —
    no next region, no on-demand prompt, no --wait sleep."""
    contexts = {
        "us-west-2": _region_ctx("us-west-2", PROFILE_REJECTED),
        "us-east-1": _region_ctx("us-east-1", [_ok_instance()]),
    }
    confirm = MagicMock(return_value=True)
    config = LaunchConfig(regions=("us-west-2", "us-east-1"), spot=True, iam_instance_profile="app", wait=wait)
    with pytest.raises(CLIError, match="rejected IAM instance profile"):
        launch_with_retry(
            config, lambda r: contexts[r], confirm_on_demand=confirm, sleeper=lambda s: pytest.fail("slept")
        )
    contexts["us-east-1"].ec2_client.run_instances.assert_not_called()
    confirm.assert_not_called()


# ---------------------------------------------------------------------------
# Instance discovery / pricing / termination
# ---------------------------------------------------------------------------


def _raw_instance(instance_id="i-abc123", *, tags=(("Name", "aws-bootstrap-g4dn.xlarge"),), **overrides):
    """An instance as describe_instances returns it."""
    inst = {
        "InstanceId": instance_id,
        "State": {"Name": "running"},
        "InstanceType": "g4dn.xlarge",
        "PublicIpAddress": "1.2.3.4",
        "LaunchTime": datetime(2025, 1, 1, tzinfo=UTC),
        "InstanceLifecycle": "spot",
        "Placement": {"AvailabilityZone": "us-west-2a"},
        "Tags": [{"Key": k, "Value": v} for k, v in tags],
        **overrides,
    }
    return {k: v for k, v in inst.items() if v is not None}


def _reservations(*instances):
    return {"Reservations": [{"Instances": [inst]} for inst in instances]}


def test_find_tagged_instances(ec2):
    ec2.describe_instances.return_value = _reservations(
        _raw_instance(tags=(("Name", "aws-bootstrap-g4dn.xlarge"), ("created-by", "aws-bootstrap-g4dn")))
    )
    instances = find_tagged_instances(ec2, "aws-bootstrap-g4dn")
    assert len(instances) == 1
    inst = instances[0]
    assert inst["InstanceId"] == "i-abc123"
    assert inst["State"] == "running"
    assert inst["PublicIp"] == "1.2.3.4"
    assert inst["Name"] == "aws-bootstrap-g4dn.xlarge"
    assert inst["Lifecycle"] == "spot"
    assert inst["AvailabilityZone"] == "us-west-2a"


def test_find_tagged_instances_on_demand_lifecycle(ec2):
    """On-demand instances have no InstanceLifecycle key; should default to 'on-demand'."""
    ec2.describe_instances.return_value = _reservations(
        _raw_instance("i-ondemand", InstanceLifecycle=None, Placement={"AvailabilityZone": "us-west-2b"})
    )
    inst = find_tagged_instances(ec2, "aws-bootstrap-g4dn")[0]
    assert inst["Lifecycle"] == "on-demand"
    assert inst["AvailabilityZone"] == "us-west-2b"


def test_find_tagged_instances_empty(ec2):
    ec2.describe_instances.return_value = {"Reservations": []}
    assert find_tagged_instances(ec2, "aws-bootstrap-g4dn") == []


@pytest.mark.parametrize(
    "tags,cluster_id,rank",
    [
        ((("Name", "n"), ("aws-bootstrap-cluster", "ml1"), ("aws-bootstrap-cluster-rank", "2")), "ml1", 2),
        ((("Name", "n"),), "", None),
    ],
    ids=["cluster-node", "standalone"],
)
def test_find_tagged_instances_exposes_cluster_tags(ec2, tags, cluster_id, rank):
    """`cleanup --sync` needs a node's cluster id/rank to rebuild its deterministic alias."""
    ec2.describe_instances.return_value = _reservations(_raw_instance("i-1", tags=tags, PublicIpAddress=None))
    inst = find_tagged_instances(ec2, "aws-bootstrap-g4dn")[0]
    assert inst["ClusterId"] == cluster_id
    assert inst["Rank"] == rank


@pytest.mark.parametrize(
    "history,expected", [([{"SpotPrice": "0.1578"}], 0.1578), ([], None)], ids=["latest-price", "no-history"]
)
def test_get_spot_price(ec2, history, expected):
    ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": history}
    assert get_spot_price(ec2, "g4dn.xlarge", "us-west-2a") == expected
    ec2.describe_spot_price_history.assert_called_once()


def test_terminate_tagged_instances(ec2):
    ec2.terminate_instances.return_value = {
        "TerminatingInstances": [
            {"InstanceId": "i-abc123", "PreviousState": {"Name": "running"}, "CurrentState": {"Name": "shutting-down"}}
        ]
    }
    changes = terminate_tagged_instances(ec2, ["i-abc123"])
    assert len(changes) == 1
    assert changes[0]["InstanceId"] == "i-abc123"
    ec2.terminate_instances.assert_called_once_with(InstanceIds=["i-abc123"])


# ---------------------------------------------------------------------------
# list_instance_types
# ---------------------------------------------------------------------------


def test_list_instance_types_returns_sorted(ec2):
    t4 = {"Gpus": [{"Count": 1, "Name": "T4", "MemoryInfo": {"SizeInMiB": 16384}}]}
    ec2.get_paginator.return_value.paginate.return_value = [
        {
            "InstanceTypes": [
                {
                    "InstanceType": "g4dn.xlarge",
                    "VCpuInfo": {"DefaultVCpus": 4},
                    "MemoryInfo": {"SizeInMiB": 16384},
                    "GpuInfo": t4,
                },
                {
                    "InstanceType": "g4dn.2xlarge",
                    "VCpuInfo": {"DefaultVCpus": 8},
                    "MemoryInfo": {"SizeInMiB": 32768},
                    "GpuInfo": t4,
                },
            ]
        }
    ]
    results = list_instance_types(ec2, "g4dn")
    # sorted by name — 2xlarge < xlarge lexicographically
    assert [r["InstanceType"] for r in results] == ["g4dn.2xlarge", "g4dn.xlarge"]
    assert results[1]["VCpuCount"] == 4
    assert results[1]["MemoryMiB"] == 16384
    assert "T4" in results[1]["GpuSummary"]


def test_list_instance_types_no_gpu(ec2):
    ec2.get_paginator.return_value.paginate.return_value = [
        {
            "InstanceTypes": [
                {"InstanceType": "t3.medium", "VCpuInfo": {"DefaultVCpus": 2}, "MemoryInfo": {"SizeInMiB": 4096}}
            ]
        }
    ]
    results = list_instance_types(ec2, "t3")
    assert len(results) == 1
    assert results[0]["GpuSummary"] == ""


@pytest.mark.parametrize(
    "prefix,expected",
    [
        # A hyphenated family (p6-b200) is only reachable via a true prefix match;
        # the old "{prefix}.*" filter returned nothing for --prefix p6.
        ("p6", "p6*.*"),
        ("g4dn", "g4dn*.*"),
        ("g", "g*.*"),
    ],
)
def test_list_instance_types_uses_prefix_match(ec2, prefix, expected):
    ec2.get_paginator.return_value.paginate.return_value = [{"InstanceTypes": []}]
    assert list_instance_types(ec2, prefix) == []
    filters = ec2.get_paginator.return_value.paginate.call_args.kwargs["Filters"]
    assert filters == [{"Name": "instance-type", "Values": [expected]}]


@pytest.mark.parametrize(
    "gpu,expected",
    [
        ({"Count": 4, "Name": "T4", "MemoryInfo": {"SizeInMiB": 16384}}, "4x T4 (16384 MiB)"),
        # Fractional GPUs (g6f/gr6f) report Count 0 and their share in GpuPartitionSize.
        (
            {
                "Count": 0,
                "LogicalGpuCount": 1,
                "GpuPartitionSize": 0.125,
                "Name": "L4",
                "MemoryInfo": {"SizeInMiB": 2861},
            },
            "1/8 L4 (2861 MiB)",
        ),
        ({"Count": 0, "GpuPartitionSize": 0.5, "Name": "L4", "MemoryInfo": {"SizeInMiB": 11444}}, "1/2 L4 (11444 MiB)"),
        ({"Name": "X"}, "?x X (0 MiB)"),
    ],
    ids=["whole-gpus", "eighth-partition", "half-partition", "missing-fields"],
)
def test_gpu_summary(gpu, expected):
    assert _gpu_summary({"Gpus": [gpu]}) == expected


def test_gpu_summary_without_gpus():
    assert _gpu_summary({}) == ""


# ---------------------------------------------------------------------------
# list_amis
# ---------------------------------------------------------------------------


def _ami(image_id, created, name=None):
    return {"ImageId": image_id, "Name": name or image_id, "CreationDate": created, "Architecture": "x86_64"}


def test_list_amis_sorted_newest_first(ec2):
    ec2.describe_images.return_value = {
        "Images": [_ami("ami-old", "2024-01-01T00:00:00Z"), _ami("ami-new", "2025-06-01T00:00:00Z")]
    }
    results = list_amis(ec2, "DL AMI*")
    assert [r["ImageId"] for r in results] == ["ami-new", "ami-old"]


@pytest.mark.parametrize(
    "architecture,arch_filter",
    [
        # No architecture filter by default — otherwise ARM64 DLAMIs are invisible.
        pytest.param(None, None, id="all-architectures-by-default"),
        pytest.param("arm64", {"Name": "architecture", "Values": ["arm64"]}, id="narrowed"),
    ],
)
def test_list_amis_architecture_filter(ec2, architecture, arch_filter):
    ec2.describe_images.return_value = {"Images": []}
    kwargs = {"architecture": architecture} if architecture else {}
    assert list_amis(ec2, "Deep Learning*", **kwargs) == []
    filters = ec2.describe_images.call_args.kwargs["Filters"]
    assert [f for f in filters if f["Name"] == "architecture"] == ([arch_filter] if arch_filter else [])


def test_list_amis_limited_to_20(ec2):
    ec2.describe_images.return_value = {
        "Images": [_ami(f"ami-{i:03d}", f"2025-01-{i + 1:02d}T00:00:00Z") for i in range(25)]
    }
    assert len(list_amis(ec2, "AMI*")) == 20


def test_list_amis_uses_owner_hint_for_deep_learning(ec2):
    ec2.describe_images.return_value = {"Images": []}
    list_amis(ec2, "Deep Learning Base*")
    assert ec2.describe_images.call_args[1]["Owners"] == ["amazon"]


# ---------------------------------------------------------------------------
# list_enabled_regions / find_tagged_instances_in_regions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names,expected",
    [(["us-west-2", "eu-west-1", "us-east-1"], ["eu-west-1", "us-east-1", "us-west-2"]), ([], [])],
    ids=["sorted", "empty"],
)
def test_list_enabled_regions(ec2, names, expected):
    ec2.describe_regions.return_value = {"Regions": [{"RegionName": n} for n in names]}
    assert list_enabled_regions(ec2) == expected
    ec2.describe_regions.assert_called_once_with(AllRegions=False)


def _region_client(instance_id: str | None = None, az: str = "", error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error:
        client.describe_instances.side_effect = error
    else:
        client.describe_instances.return_value = _reservations(
            _raw_instance(instance_id, Placement={"AvailabilityZone": az}, tags=(("created-by", "aws-bootstrap-g4dn"),))
        )
    return client


def test_find_tagged_instances_in_regions_aggregates_and_labels():
    session = session_with_clients(
        {"us-east-1": _region_client("i-east", "us-east-1a"), "us-west-2": _region_client("i-west", "us-west-2a")}
    )
    instances, failures = find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", ["us-east-1", "us-west-2"])
    assert failures == []
    by_id = {i["InstanceId"]: i for i in instances}
    assert set(by_id) == {"i-east", "i-west"}
    assert by_id["i-east"]["Region"] == "us-east-1"
    assert by_id["i-west"]["Region"] == "us-west-2"
    # Sorted by region then launch time
    assert [i["Region"] for i in instances] == ["us-east-1", "us-west-2"]


def test_find_tagged_instances_in_regions_captures_failure():
    bad = _region_client(error=client_error("AuthFailure"))
    session = session_with_clients({"us-east-1": _region_client("i-ok", "us-east-1a"), "ap-south-1": bad})
    instances, failures = find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", ["us-east-1", "ap-south-1"])
    assert [i["InstanceId"] for i in instances] == ["i-ok"]
    assert len(failures) == 1
    assert failures[0]["region"] == "ap-south-1"
    assert "AuthFailure" in failures[0]["error"]


def test_find_tagged_instances_in_regions_empty_region_list():
    session = MagicMock()
    assert find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", []) == ([], [])
    session.client.assert_not_called()


def test_find_tagged_instances_in_regions_propagates_credential_error():
    """Non-ClientError exceptions (e.g. missing credentials) are not swallowed."""
    session = session_with_clients({"us-east-1": _region_client(error=botocore.exceptions.NoCredentialsError())})
    with pytest.raises(botocore.exceptions.NoCredentialsError):
        find_tagged_instances_in_regions(session, "aws-bootstrap-g4dn", ["us-east-1"])


# ---------------------------------------------------------------------------
# Cluster placement-group lifecycle / intra-cluster SG rule / discovery
# ---------------------------------------------------------------------------


def test_ensure_cluster_placement_group_creates_when_absent(ec2):
    ec2.describe_placement_groups.return_value = {"PlacementGroups": []}
    assert ensure_cluster_placement_group(ec2, "aws-bootstrap-cluster-ml1", "aws-bootstrap-g4dn") == (
        "aws-bootstrap-cluster-ml1"
    )
    create_kwargs = ec2.create_placement_group.call_args[1]
    assert create_kwargs["GroupName"] == "aws-bootstrap-cluster-ml1"
    assert create_kwargs["Strategy"] == "cluster"


def test_ensure_cluster_placement_group_reuses_existing(ec2):
    ec2.describe_placement_groups.return_value = {
        "PlacementGroups": [{"GroupName": "aws-bootstrap-cluster-ml1", "State": "available"}]
    }
    assert ensure_cluster_placement_group(ec2, "aws-bootstrap-cluster-ml1", "aws-bootstrap-g4dn") == (
        "aws-bootstrap-cluster-ml1"
    )
    ec2.create_placement_group.assert_not_called()


@pytest.mark.parametrize(
    "error,deleted",
    [
        pytest.param(None, True, id="deleted"),
        # Already gone counts as deleted.
        pytest.param("InvalidPlacementGroup.Unknown", True, id="already-gone"),
        # Instances still terminating -> not deletable yet, but no raise.
        pytest.param("InvalidPlacementGroup.InUse", False, id="in-use"),
    ],
)
def test_delete_cluster_placement_group(ec2, error, deleted):
    if error:
        ec2.delete_placement_group.side_effect = client_error(error)
    assert delete_cluster_placement_group(ec2, "aws-bootstrap-cluster-ml1") is deleted
    ec2.delete_placement_group.assert_called_once_with(GroupName="aws-bootstrap-cluster-ml1")


def test_ensure_cluster_sg_rule_authorizes_self_reference(ec2):
    ensure_cluster_security_group_rule(ec2, "sg-123")
    kwargs = ec2.authorize_security_group_ingress.call_args[1]
    assert kwargs["GroupId"] == "sg-123"
    perm = kwargs["IpPermissions"][0]
    assert perm["IpProtocol"] == "-1"
    assert perm["UserIdGroupPairs"][0]["GroupId"] == "sg-123"


def test_ensure_cluster_sg_rule_idempotent_on_duplicate(ec2):
    ec2.authorize_security_group_ingress.side_effect = client_error("InvalidPermission.Duplicate")
    ensure_cluster_security_group_rule(ec2, "sg-123")  # must not raise


def _cluster_reservation(instance_id, cluster_id, rank, private="10.0.0.5"):
    return _reservations(
        _raw_instance(
            instance_id,
            InstanceType="g5.xlarge",
            PrivateIpAddress=private,
            LaunchTime=datetime(2026, 5, 21, tzinfo=UTC),
            Placement={"AvailabilityZone": "us-east-1c"},
            tags=(("aws-bootstrap-cluster", cluster_id), ("aws-bootstrap-cluster-rank", str(rank))),
        )
    )["Reservations"][0]


def test_find_cluster_instances_sorted_by_rank(ec2):
    ec2.describe_instances.return_value = {
        "Reservations": [
            _cluster_reservation("i-2", "ml1", 1, private="10.0.0.6"),
            _cluster_reservation("i-1", "ml1", 0, private="10.0.0.5"),
        ]
    }
    nodes = find_cluster_instances(ec2, "ml1")
    assert [n["Rank"] for n in nodes] == [0, 1]
    assert nodes[0]["InstanceId"] == "i-1"
    assert nodes[0]["PrivateIp"] == "10.0.0.5"
    assert nodes[0]["AvailabilityZone"] == "us-east-1c"
    filters = ec2.describe_instances.call_args[1]["Filters"]
    assert {"Name": "tag:aws-bootstrap-cluster", "Values": ["ml1"]} in filters


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param({}, "", id="none"),
        pytest.param({"IamInstanceProfile": {"Arn": PROFILE_ARN, "Id": "AIPA123"}}, "app", id="attached"),
    ],
)
def test_instance_discovery_reports_the_profile_name(ec2, raw, expected):
    """describe_instances gives only the profile's ARN; status and clusters show its name."""
    ec2.describe_instances.return_value = _reservations(_raw_instance(**raw))
    assert find_tagged_instances(ec2, "aws-bootstrap-g4dn")[0]["IamInstanceProfile"] == expected
    assert find_cluster_instances(ec2, "ml1")[0]["IamInstanceProfile"] == expected


def test_list_clusters_groups_by_cluster_id(ec2):
    ec2.describe_instances.return_value = {
        "Reservations": [
            _cluster_reservation("i-1", "ml1", 0),
            _cluster_reservation("i-2", "ml1", 1),
            _cluster_reservation("i-3", "exp2", 0),
        ]
    }
    clusters = list_clusters(ec2, "aws-bootstrap-g4dn")
    assert set(clusters) == {"ml1", "exp2"}
    assert len(clusters["ml1"]) == 2
    assert len(clusters["exp2"]) == 1


# ---------------------------------------------------------------------------
# Security-group SSH ingress on reuse
# ---------------------------------------------------------------------------


def _sg_client(permissions, *, owned=True):
    """A client whose SG lookup hits an existing group. ``owned`` controls whether
    it carries this tool's created-by tag — the gate on mutating its ingress."""
    ec2 = MagicMock()
    ec2.describe_vpcs.return_value = {"Vpcs": [{"VpcId": "vpc-1"}]}
    tags = [{"Key": TAG_CREATED_BY, "Value": "tv"}] if owned else [{"Key": "Name", "Value": "someone-elses-sg"}]
    ec2.describe_security_groups.return_value = {
        "SecurityGroups": [{"GroupId": "sg-1", "IpPermissions": permissions, "Tags": tags}]
    }
    return ec2


def _tcp_rule(from_port, to_port, cidr="0.0.0.0/0"):
    return [{"IpProtocol": "tcp", "FromPort": from_port, "ToPort": to_port, "IpRanges": [{"CidrIp": cidr}]}]


_SSH_22 = _tcp_rule(22, 22)


@pytest.mark.parametrize(
    "permissions,port",
    [
        # Regression: --ssh-port against an existing SG left it with no ingress,
        # surfacing only much later as an opaque 'SSH never became available'.
        pytest.param(_SSH_22, 2222, id="missing-port"),
        # A rule open only to some other CIDR does not make our port reachable.
        pytest.param(_tcp_rule(22, 22, "10.0.0.0/8"), 22, id="other-cidr-only"),
    ],
)
def test_reused_owned_sg_opens_an_unreachable_port(permissions, port):
    ec2 = _sg_client(permissions)
    assert ensure_security_group(ec2, "aws-bootstrap-ssh", "tv", ssh_port=port) == "sg-1"
    ec2.authorize_security_group_ingress.assert_called_once()
    perm = ec2.authorize_security_group_ingress.call_args.kwargs["IpPermissions"][0]
    assert perm["FromPort"] == perm["ToPort"] == port


@pytest.mark.parametrize(
    "permissions,port",
    [pytest.param(_SSH_22, 22, id="exact-rule"), pytest.param(_tcp_rule(20, 30), 22, id="port-range-covers-it")],
)
def test_reused_owned_sg_does_not_re_add_an_existing_rule(permissions, port):
    ec2 = _sg_client(permissions)
    assert ensure_security_group(ec2, "aws-bootstrap-ssh", "tv", ssh_port=port) == "sg-1"
    ec2.authorize_security_group_ingress.assert_not_called()


def test_reused_sg_never_widens_a_group_this_tool_did_not_create():
    """`--security-group` is user-supplied and the lookup matches on name + VPC
    alone, so a pre-existing restricted group can share the name. Punching
    0.0.0.0/0 into it would be a security change nobody asked for."""
    ec2 = _sg_client(_tcp_rule(22, 22, "10.0.0.0/8"), owned=False)
    assert ensure_security_group(ec2, "someone-elses-sg", "tv", ssh_port=2222) == "sg-1"
    ec2.authorize_security_group_ingress.assert_not_called()


def test_unowned_sg_that_already_has_ingress_is_silent(capsys):
    """Nothing to do and nothing to warn about — the port is already reachable."""
    ec2 = _sg_client(_SSH_22, owned=False)
    assert ensure_security_group(ec2, "someone-elses-sg", "tv", ssh_port=22) == "sg-1"
    ec2.authorize_security_group_ingress.assert_not_called()
    assert "not tagged" not in capsys.readouterr().err


def test_reused_sg_tolerates_a_concurrent_duplicate_add():
    ec2 = _sg_client([])
    ec2.authorize_security_group_ingress.side_effect = client_error("InvalidPermission.Duplicate")
    assert ensure_security_group(ec2, "aws-bootstrap-ssh", "tv", ssh_port=22) == "sg-1"


def test_reused_sg_propagates_an_unexpected_authorize_error():
    ec2 = _sg_client([])
    ec2.authorize_security_group_ingress.side_effect = client_error("UnauthorizedOperation")
    with pytest.raises(botocore.exceptions.ClientError):
        ensure_security_group(ec2, "aws-bootstrap-ssh", "tv", ssh_port=22)


def test_unowned_sg_without_ingress_warns_with_the_exact_command(capsys):
    ensure_security_group(_sg_client([], owned=False), "someone-elses-sg", "tv", ssh_port=2222)
    err = capsys.readouterr().err
    assert "not tagged" in err
    assert "authorize-security-group-ingress --group-id sg-1" in err
    assert "--port 2222" in err


# ---------------------------------------------------------------------------
# Terminate ownership guard
# ---------------------------------------------------------------------------


def _describe_owned(*pages):
    """A client whose tag-filtered describe_instances pages hold the given owned ids."""
    ec2 = MagicMock()
    ec2.get_paginator.return_value.paginate.return_value = [
        {"Reservations": [{"Instances": [{"InstanceId": i} for i in page]}]} for page in pages
    ]
    return ec2


@pytest.mark.parametrize(
    "pages,query,unowned",
    [
        pytest.param([["i-mine"]], ["i-mine", "i-theirs"], ["i-theirs"], id="flags-untagged"),
        pytest.param([["i-a", "i-b"]], ["i-a", "i-b"], [], id="all-owned"),
        # Nothing tagged in this region -> every id is unowned (likely wrong --region).
        pytest.param([[]], ["i-a"], ["i-a"], id="absent-from-region"),
        # A truncated page would misclassify an owned instance as unowned and
        # refuse a legitimate terminate.
        pytest.param([["i-page1"], ["i-page2"]], ["i-page1", "i-page2"], [], id="paginates"),
    ],
)
def test_find_unowned_instances(pages, query, unowned):
    ec2 = _describe_owned(*pages)
    assert find_unowned_instances(ec2, query, TAG_VALUE) == unowned
    ec2.get_paginator.assert_called_once_with("describe_instances")


def test_find_unowned_instances_short_circuits_on_empty_input(ec2):
    assert find_unowned_instances(ec2, [], TAG_VALUE) == []
    ec2.get_paginator.assert_not_called()


def test_find_unowned_instances_filters_by_the_created_by_tag():
    ec2 = _describe_owned(["i-a"])
    find_unowned_instances(ec2, ["i-a"], TAG_VALUE)
    kwargs = ec2.get_paginator.return_value.paginate.call_args.kwargs
    assert kwargs["Filters"] == [{"Name": f"tag:{TAG_CREATED_BY}", "Values": [TAG_VALUE]}]
    # Passing InstanceIds would make AWS raise for any id it has never seen,
    # which is precisely the case this check exists to answer.
    assert "InstanceIds" not in kwargs
