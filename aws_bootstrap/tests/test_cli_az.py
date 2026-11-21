"""CLI: --availability-zone / --cheapest-az on launch and cluster launch, and the pre-launch key check."""

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest
import yaml
from click.testing import CliRunner

from aws_bootstrap.cli import _emit_region_fatal, main
from aws_bootstrap.ec2 import AzOption, CLIError, RegionContext, RegionLaunch


PLAN_USW2 = [AzOption("us-west-2b", 0.9911), AzOption("us-west-2a", 2.2122)]
PLAN_USE1 = [AzOption("us-east-1c", 2.0426), AzOption("us-east-1a", 2.0539)]
PLANS = {"us-west-2": PLAN_USW2, "us-east-1": PLAN_USE1}


def _run(*args: str):
    return CliRunner().invoke(main, list(args))


@pytest.fixture
def key(tmp_path):
    pub = tmp_path / "id_ed25519.pub"
    pub.write_text("ssh-ed25519 AAAA test@host")
    return pub


@pytest.fixture
def launch_aws(cli_session):
    """The AWS-touching helpers a launch prepares with (dry-run or not); yields the mocks."""
    with (
        patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-1", "Name": "DLAMI"}),
        patch("aws_bootstrap.cli.instance_type_architecture", return_value="x86_64"),
        patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key") as import_key,
        patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-1"),
        patch("aws_bootstrap.cli.plan_azs") as plan,
    ):
        plan.side_effect = lambda ec2, region, *a, **kw: PLANS[region]
        yield {"plan": plan, "import_key": import_key}


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
)
def test_launch_rejects_bad_az_flags_before_any_aws_call(args, message, key):
    with patch("aws_bootstrap.cli.boto3.Session") as session:
        result = _run("launch", "--key-path", str(key), "--dry-run", *args)
    assert result.exit_code != 0
    assert message in result.output
    session.assert_not_called()


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_az_implies_its_region_and_is_planned_explicitly(launch_aws, key):
    result = _run("-o", "json", "launch", "--key-path", str(key), "--dry-run", "--az", "us-west-2b")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["regions"] == ["us-west-2"]
    plan = launch_aws["plan"]
    assert plan.call_args.args[1:] == ("us-west-2", "g4dn.xlarge")
    assert plan.call_args.kwargs == {"explicit": ("us-west-2b",), "cheapest": False}


def test_dry_run_plans_every_region_not_only_the_first(launch_aws, key):
    regions = ["--region", "us-west-2", "--region", "us-east-1"]
    result = _run("-o", "json", "launch", "--key-path", str(key), "--dry-run", "--cheapest-az", *regions)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["availability_zones"] == {
        r: [{"zone": o.zone, "spot_price": o.spot_price} for o in plan] for r, plan in PLANS.items()
    }


@pytest.mark.parametrize("fmt", ["text", "json", "yaml", "table"])
def test_dry_run_az_plan_renders_in_every_output_format(launch_aws, key, fmt):
    result = _run("-o", fmt, "launch", "--key-path", str(key), "--dry-run", "--cheapest-az", "--region", "us-west-2")
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


def test_dry_run_without_az_flags_reports_null(launch_aws, key):
    result = _run("-o", "json", "launch", "--key-path", str(key), "--dry-run")
    assert json.loads(result.output)["availability_zones"] is None
    launch_aws["plan"].assert_not_called()


def test_bad_az_in_a_later_region_fails_before_anything_is_created(launch_aws, key):
    """Every region is planned up front: region 2's bad AZ must not surface after region 1 launched."""

    def plan(ec2, region, *a, **kw):
        if region == "us-east-1":
            raise CLIError("Unknown availability zone in us-east-1: us-east-1z")
        return PLAN_USW2

    launch_aws["plan"].side_effect = plan
    with patch("aws_bootstrap.cli.launch_with_retry") as launch:
        result = _run("launch", "--key-path", str(key), "--az", "us-west-2b", "--az", "us-east-1z")
    assert result.exit_code != 0
    assert "[us-east-1] Unknown availability zone" in result.output
    launch.assert_not_called()
    launch_aws["import_key"].assert_not_called()


def test_prepare_region_threads_the_plan_into_the_context(launch_aws, key):
    captured: dict = {}

    def fake_launch(config, prepare_region, **kw):
        captured["ctx"] = prepare_region("us-west-2")
        raise CLIError("stop here")

    with patch("aws_bootstrap.cli.launch_with_retry", side_effect=fake_launch):
        _run("launch", "--key-path", str(key), "--cheapest-az", "--region", "us-west-2")
    assert captured["ctx"].placement_azs == ("us-west-2b", "us-west-2a")
    assert captured["ctx"].az_prices == {"us-west-2b": 0.9911, "us-west-2a": 2.2122}


# ---------------------------------------------------------------------------
# real launch: chosen AZ + its spot price reported, never fatal
# ---------------------------------------------------------------------------


def _launched(az_prices: dict, describe_spot) -> RegionLaunch:
    """A RegionLaunch in us-west-2b whose client answers describe_spot_price_history with *describe_spot*."""
    client = MagicMock()
    if isinstance(describe_spot, Exception):
        client.describe_spot_price_history.side_effect = describe_spot
    else:
        client.describe_spot_price_history.return_value = {"SpotPriceHistory": [{"SpotPrice": describe_spot}]}
    ctx = RegionContext(
        region="us-west-2",
        ec2_client=client,
        ami={"ImageId": "ami-1", "Name": "DLAMI"},
        sg_id="sg-1",
        key_name="k",
        az_prices=az_prices,
    )
    instance = {"InstanceId": "i-1", "Placement": {"AvailabilityZone": "us-west-2b"}}
    return RegionLaunch("us-west-2", ctx, instance, "spot")


THROTTLED = botocore.exceptions.ClientError({"Error": {"Code": "RequestLimitExceeded", "Message": "x"}}, "Op")


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
def test_launch_reports_the_az_and_its_spot_price(key, az_prices, describe_spot, ssh_ok, expected_price):
    with (
        patch("aws_bootstrap.cli.launch_with_retry", return_value=_launched(az_prices, describe_spot)),
        patch(
            "aws_bootstrap.cli.wait_instance_ready",
            return_value={"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2b"}},
        ),
        patch("aws_bootstrap.cli.wait_for_ssh", return_value=ssh_ok),
        patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1"),
    ):
        result = _run("-o", "json", "launch", "--key-path", str(key), "--no-setup")
    data = json.loads(result.output)
    assert result.exit_code == (0 if ssh_ok else 1)
    assert data["status"] == ("ready" if ssh_ok else "ssh-unavailable")
    assert data["instance_id"] == "i-1"
    assert data["availability_zone"] == "us-west-2b"
    assert data.get("spot_price_per_hour") == expected_price


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
def test_unusable_key_fails_before_any_aws_client(key, dry_run):
    """key is a .pub with no private key next to it (and no agent): the real check stops the launch first."""
    with (
        patch("aws_bootstrap.ssh._agent_has_key", return_value=False),
        patch("aws_bootstrap.cli.boto3.Session") as session,
        patch("aws_bootstrap.cli.launch_with_retry") as launch,
    ):
        result = _run("launch", "--key-path", str(key), *(["--dry-run"] if dry_run else []))
    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    # A Session may be built to read the profile's region (local config); no AWS client is.
    session.return_value.client.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("extra,calls", [([], 1), (["--skip-key-check"], 0)])
def test_key_check_runs_on_dry_run_unless_skipped(key_check_stub, launch_aws, key, extra, calls):
    result = _run("launch", "--key-path", str(key), "--dry-run", *extra)
    assert result.exit_code == 0, result.output
    assert key_check_stub.call_count == calls


# ---------------------------------------------------------------------------
# cluster launch
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_aws(cli_session):
    with (
        patch("aws_bootstrap.cli.find_cluster_instances", return_value=[]) as existing,
        patch("aws_bootstrap.cli.ensure_cluster_placement_group") as pg,
        patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-1"),
        patch("aws_bootstrap.cli.ensure_cluster_security_group_rule"),
        patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-1"}),
        patch("aws_bootstrap.cli.instance_type_architecture", return_value="x86_64"),
        patch("aws_bootstrap.cli.import_key_pair", return_value="k"),
        patch("aws_bootstrap.cli.plan_azs", return_value=PLAN_USW2) as plan,
        patch("aws_bootstrap.cli.cluster_mod.launch_cluster_nodes") as nodes,
    ):
        yield {"existing": existing, "pg": pg, "plan": plan, "nodes": nodes}


def _cluster(key, *extra):
    return _run("cluster", "launch", "--cluster-id", "c1", "--key-path", str(key), "--no-setup", *extra)


def test_cluster_cheapest_az_plans_the_first_node(cluster_aws, key):
    captured: dict = {}
    cluster_aws["nodes"].side_effect = lambda config, prepare_region, *a, **kw: captured.update(
        ctx=prepare_region("us-west-2")
    )
    result = _cluster(key, "--cheapest-az", "--region", "us-west-2")
    assert result.exit_code == 0, result.output
    assert cluster_aws["plan"].call_args.kwargs == {"explicit": (), "cheapest": True}
    assert captured["ctx"].placement_azs == ("us-west-2b", "us-west-2a")


def test_cluster_az_implies_region(cluster_aws, key):
    result = _cluster(key, "--az", "us-east-1c")
    assert result.exit_code == 0, result.output
    assert cluster_aws["plan"].call_args.args[1] == "us-east-1"


def test_cluster_existing_az_conflict_is_an_error(cluster_aws, key):
    cluster_aws["existing"].return_value = [{"AvailabilityZone": "us-west-2a", "Rank": 0, "InstanceId": "i-0"}]
    result = _cluster(key, "--nodes", "3", "--az", "us-west-2b")
    assert result.exit_code != 0
    assert "is in us-west-2a" in result.output
    cluster_aws["nodes"].assert_not_called()


def test_cluster_bad_az_creates_nothing(cluster_aws, key):
    """The AZ plan is read-only and runs before the placement group / SG rule are created."""
    cluster_aws["plan"].side_effect = CLIError("Unknown availability zone in us-west-2: us-west-2z")
    result = _cluster(key, "--az", "us-west-2z")
    assert result.exit_code != 0
    cluster_aws["pg"].assert_not_called()


@pytest.mark.real_key_check
def test_cluster_unusable_key_fails_before_creating_anything(cluster_aws, key):
    with patch("aws_bootstrap.ssh._agent_has_key", return_value=False):
        result = _cluster(key)
    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    cluster_aws["pg"].assert_not_called()
    cluster_aws["nodes"].assert_not_called()
