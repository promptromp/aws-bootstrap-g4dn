"""CLI: --availability-zone / --cheapest-az on launch and cluster launch, and the pre-launch key check."""

from __future__ import annotations
import json
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from aws_bootstrap.cli import _emit_region_fatal, main
from aws_bootstrap.ec2 import AzOption, CLIError
from aws_bootstrap.tests.helpers import client_error


PLAN_USW2 = [AzOption("us-west-2b", 0.9911), AzOption("us-west-2a", 2.2122)]
PLAN_USE1 = [AzOption("us-east-1c", 2.0426), AzOption("us-east-1a", 2.0539)]
PLANS = {"us-west-2": PLAN_USW2, "us-east-1": PLAN_USE1}


def _run(*args: str):
    return CliRunner().invoke(main, list(args))


@pytest.fixture
def plan_azs(launch_aws):
    """The conftest launch setup plus ``plan_azs`` answering from PLANS; yields the plan_azs mock."""
    with patch("aws_bootstrap.cli.plan_azs", side_effect=lambda ec2, region, *a, **kw: PLANS[region]) as plan:
        yield plan


# ---------------------------------------------------------------------------
# flag validation (no AWS calls)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,message",
    [
        (["--cheapest-az", "--on-demand"], "on-demand prices don't vary by AZ"),
        (["--az", "us-west-2b", "--ebs-volume-id", "vol-1"], "the volume already pins"),
        (["--cheapest-az", "--ebs-volume-id", "vol-1"], "the volume already pins"),
        (["--az", "us-east-1c", "--region", "us-west-2"], "us-east-1c is not in any --region"),
        (["--az", "uswest2b"], "Not an availability zone"),
    ],
    ids=["cheapest-on-demand", "az-with-ebs-volume", "cheapest-with-ebs-volume", "az-outside-regions", "malformed-az"],
)
def test_launch_rejects_bad_az_flags_before_any_aws_call(args, message, key_path):
    with patch("aws_bootstrap.cli.boto3.Session") as session:
        result = _run("launch", "--key-path", str(key_path), "--dry-run", *args)
    assert result.exit_code != 0
    assert message in result.output
    session.assert_not_called()


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_az_implies_its_region_and_is_planned_explicitly(plan_azs, key_path):
    result = _run("-o", "json", "launch", "--key-path", str(key_path), "--dry-run", "--az", "us-west-2b")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["regions"] == ["us-west-2"]
    assert plan_azs.call_args.args[1:] == ("us-west-2", "g4dn.xlarge")
    assert plan_azs.call_args.kwargs == {"explicit": ("us-west-2b",), "cheapest": False}


def test_dry_run_plans_every_region_not_only_the_first(plan_azs, key_path):
    regions = ["--region", "us-west-2", "--region", "us-east-1"]
    result = _run("-o", "json", "launch", "--key-path", str(key_path), "--dry-run", "--cheapest-az", *regions)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["availability_zones"] == {
        r: [{"zone": o.zone, "spot_price": o.spot_price} for o in options] for r, options in PLANS.items()
    }


@pytest.mark.parametrize("fmt", ["text", "json", "yaml", "table"])
def test_dry_run_az_plan_renders_in_every_output_format(plan_azs, key_path, fmt):
    result = _run(
        "-o", fmt, "launch", "--key-path", str(key_path), "--dry-run", "--cheapest-az", "--region", "us-west-2"
    )
    assert result.exit_code == 0, result.output
    out = result.output
    if fmt in ("json", "yaml"):
        data = json.loads(out) if fmt == "json" else yaml.safe_load(out)
        assert data["availability_zones"]["us-west-2"][0]["zone"] == "us-west-2b"
    elif fmt == "table":
        assert "us-west-2: us-west-2b $0.991/h → us-west-2a $2.212/h" in out
        assert "{" not in out  # no Python repr of a nested mapping
    else:
        assert "us-west-2b $0.991/h → us-west-2a $2.212/h" in out


def test_dry_run_without_az_flags_reports_null(plan_azs, key_path):
    result = _run("-o", "json", "launch", "--key-path", str(key_path), "--dry-run")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["availability_zones"] is None
    plan_azs.assert_not_called()


def test_bad_az_in_a_later_region_fails_before_anything_is_created(launch_aws, plan_azs, key_path):
    """Every region is planned up front: region 2's bad AZ must not surface after region 1 launched."""

    def bad_az_in_east(ec2, region, *a, **kw):
        if region == "us-east-1":
            raise CLIError("Unknown availability zone in us-east-1: us-east-1z")
        return PLAN_USW2

    plan_azs.side_effect = bad_az_in_east
    with patch("aws_bootstrap.cli.launch_with_retry") as launch:
        result = _run("launch", "--key-path", str(key_path), "--az", "us-west-2b", "--az", "us-east-1z")
    assert result.exit_code != 0
    assert "[us-east-1] Unknown availability zone" in result.output
    launch.assert_not_called()
    launch_aws.import_key.assert_not_called()


def test_cheapest_only_region_that_cannot_host_the_type_is_skipped(plan_azs, key_path):
    """Without an explicit AZ there, a region with no usable AZ is skipped (warned), not fatal."""

    def no_east(ec2, region, *a, **kw):
        if region == "us-east-1":
            raise CLIError("g6e.2xlarge is not offered in any availability zone of us-east-1.")
        return PLAN_USW2

    plan_azs.side_effect = no_east
    regions = ["--region", "us-east-1", "--region", "us-west-2"]
    result = _run("-o", "json", "launch", "--key-path", str(key_path), "--dry-run", "--cheapest-az", *regions)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["regions"] == ["us-west-2"]
    assert list(data["availability_zones"]) == ["us-west-2"]
    assert "[us-east-1]" in result.stderr and "Skipping us-east-1" in result.stderr  # visible in -o json too


def test_no_region_can_host_the_launch(plan_azs, key_path):
    plan_azs.side_effect = CLIError("not offered in any availability zone")
    result = _run("launch", "--key-path", str(key_path), "--dry-run", "--cheapest-az", "--region", "us-west-2")
    assert result.exit_code != 0
    assert "No --region can host this launch" in result.output


def test_prepare_region_threads_the_plan_into_the_context(plan_azs, key_path):
    captured: dict = {}

    def fake_launch(config, prepare_region, **kw):
        captured["ctx"] = prepare_region("us-west-2")
        raise CLIError("stop here")

    with patch("aws_bootstrap.cli.launch_with_retry", side_effect=fake_launch):
        _run("launch", "--key-path", str(key_path), "--cheapest-az", "--region", "us-west-2")
    assert captured["ctx"].placement_azs == ("us-west-2b", "us-west-2a")
    assert captured["ctx"].az_prices == {"us-west-2b": 0.9911, "us-west-2a": 2.2122}
    assert captured["ctx"].hard_pin is False  # a plan, not a required AZ


def test_ebs_volume_az_is_a_hard_pin(plan_azs, key_path):
    captured: dict = {}

    def fake_launch(config, prepare_region, **kw):
        captured["ctx"] = prepare_region("us-west-2")
        raise CLIError("stop here")

    with patch("aws_bootstrap.cli.launch_with_retry", side_effect=fake_launch):
        _run("launch", "--key-path", str(key_path), "--ebs-volume-id", "vol-1", "--region", "us-west-2")
    assert captured["ctx"].placement_azs == ("us-west-2a",)  # the located volume's AZ (conftest)
    assert captured["ctx"].hard_pin is True


# ---------------------------------------------------------------------------
# real launch: chosen AZ + its spot price reported, never fatal
# ---------------------------------------------------------------------------


THROTTLED = client_error("RequestLimitExceeded")


@pytest.mark.parametrize(
    "az_prices,describe_spot,ssh_ok,expected_price",
    [
        ({"us-west-2b": 0.9911}, "0.95", True, 0.95),  # a fresh lookup beats the (possibly stale) plan price
        ({"us-west-2b": 0.9911}, THROTTLED, True, 0.9911),  # lookup fails: fall back to the plan
        ({}, THROTTLED, True, None),  # unknown: omitted, never fatal
        ({}, THROTTLED, False, None),  # the incomplete-launch record still carries the instance
    ],
    ids=["fresh", "plan-fallback", "unknown", "ssh-unavailable"],
)
def test_launch_reports_the_az_and_its_spot_price(
    launched, region_launch, key_path, az_prices, describe_spot, ssh_ok, expected_price
):
    """The real spot_price_or_none runs against the launch's own client (describe_spot_price_history)."""
    landed = region_launch({"InstanceId": "i-1", "Placement": {"AvailabilityZone": "us-west-2b"}}, az_prices=az_prices)
    spot_history = landed.context.ec2_client.describe_spot_price_history
    if isinstance(describe_spot, Exception):
        spot_history.side_effect = describe_spot
    else:
        spot_history.return_value = {"SpotPriceHistory": [{"SpotPrice": describe_spot}]}
    launched.launch.return_value = landed
    launched.ready.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2b"}}
    launched.ssh_ready.return_value = ssh_ok
    result = _run("-o", "json", "launch", "--key-path", str(key_path), "--no-setup")
    data = json.loads(result.output)
    assert result.exit_code == (0 if ssh_ok else 1)
    assert data["status"] == ("ready" if ssh_ok else "ssh-unavailable")
    assert data["instance_id"] == "i-1"
    assert data["availability_zone"] == "us-west-2b"
    assert data.get("spot_price_per_hour") == expected_price
    spot_history.assert_called_once()  # the price came from the landed region's own client
    launched.setup.assert_not_called()  # --no-setup


@pytest.mark.parametrize(
    "kind,label",
    [
        ("quota", "spot quota exceeded"),
        ("price", "spot price too low"),
        ("unsupported", "launch not supported in the requested AZ(s)"),
    ],
)
def test_region_fatal_labels(capsys, kind, label):
    _emit_region_fatal("us-west-2", kind, "details", more_regions=False)
    assert f"us-west-2: {label}" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# pre-launch key check wiring
# ---------------------------------------------------------------------------


@pytest.mark.real_key_check
@pytest.mark.parametrize("dry_run", [True, False])
def test_unusable_key_fails_before_any_aws_client(key_path, dry_run):
    """key_path is a .pub with no private key next to it (and no agent): the real check stops the launch first."""
    with (
        patch("aws_bootstrap.ssh._agent_has_key", return_value=False),
        patch("aws_bootstrap.cli.boto3.Session") as session,
        patch("aws_bootstrap.cli.launch_with_retry") as launch,
    ):
        result = _run("launch", "--key-path", str(key_path), *(["--dry-run"] if dry_run else []))
    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    # A Session may be built to read the profile's region (local config); no AWS client is.
    session.return_value.client.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("extra,calls", [([], 1), (["--skip-key-check"], 0)], ids=["checked", "skipped"])
def test_key_check_runs_on_dry_run_unless_skipped(key_check_stub, plan_azs, key_path, extra, calls):
    result = _run("launch", "--key-path", str(key_path), "--dry-run", *extra)
    assert result.exit_code == 0, result.output
    assert key_check_stub.call_count == calls


# ---------------------------------------------------------------------------
# cluster launch
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_aws(plan_azs):
    """Cluster launch on top of the conftest launch setup (Session, AMI, arch, key_path, SG) and ``plan_azs``."""
    with (
        patch("aws_bootstrap.cli.find_cluster_instances", return_value=[]) as existing,
        patch("aws_bootstrap.cli.ensure_cluster_placement_group") as pg,
        patch("aws_bootstrap.cli.ensure_cluster_security_group_rule"),
        patch("aws_bootstrap.cli.cluster_mod.launch_cluster_nodes") as nodes,
    ):
        yield {"existing": existing, "pg": pg, "plan": plan_azs, "nodes": nodes}


def _cluster(key_path, *extra):
    return _run("cluster", "launch", "--cluster-id", "c1", "--key-path", str(key_path), "--no-setup", *extra)


def test_cluster_cheapest_az_plans_the_first_node(cluster_aws, key_path):
    captured: dict = {}
    cluster_aws["nodes"].side_effect = lambda config, prepare_region, *a, **kw: captured.update(
        ctx=prepare_region("us-west-2")
    )
    result = _cluster(key_path, "--cheapest-az", "--region", "us-west-2")
    assert result.exit_code == 0, result.output
    assert cluster_aws["plan"].call_args.kwargs == {"explicit": (), "cheapest": True}
    assert captured["ctx"].placement_azs == ("us-west-2b", "us-west-2a")


def test_cluster_az_implies_region(cluster_aws, key_path):
    result = _cluster(key_path, "--az", "us-east-1c")
    assert result.exit_code == 0, result.output
    assert cluster_aws["plan"].call_args.args[1] == "us-east-1"


def test_cluster_existing_az_conflict_is_an_error(cluster_aws, key_path):
    cluster_aws["existing"].return_value = [{"AvailabilityZone": "us-west-2a", "Rank": 0, "InstanceId": "i-0"}]
    result = _cluster(key_path, "--nodes", "3", "--az", "us-west-2b")
    assert result.exit_code != 0
    assert "is in us-west-2a" in result.output
    cluster_aws["nodes"].assert_not_called()


def test_cluster_bad_az_creates_nothing(cluster_aws, key_path):
    """The AZ plan is read-only and runs before the placement group / SG rule are created."""
    cluster_aws["plan"].side_effect = CLIError("Unknown availability zone in us-west-2: us-west-2z")
    result = _cluster(key_path, "--az", "us-west-2z")
    assert result.exit_code != 0
    cluster_aws["pg"].assert_not_called()


@pytest.mark.real_key_check
def test_cluster_unusable_key_fails_before_creating_anything(cluster_aws, key_path):
    with patch("aws_bootstrap.ssh._agent_has_key", return_value=False):
        result = _cluster(key_path)
    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    cluster_aws["pg"].assert_not_called()
    cluster_aws["nodes"].assert_not_called()
