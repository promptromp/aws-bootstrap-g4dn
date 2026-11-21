"""CLI: --availability-zone / --cheapest-az on launch and cluster launch, and the pre-launch key check."""

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.ec2 import AzOption, CLIError, RegionContext, RegionLaunch


PLAN_USW2 = [AzOption("us-west-2b", 0.9911), AzOption("us-west-2a", 2.2122)]
PLAN_USE1 = [AzOption("us-east-1c", 2.0426), AzOption("us-east-1a", 2.0539)]


@pytest.fixture
def key(tmp_path):
    pub = tmp_path / "id_ed25519.pub"
    pub.write_text("ssh-ed25519 AAAA test@host")
    return pub


@pytest.fixture
def dry_run_aws():
    """Patch the AWS-touching helpers a dry run uses; yields the plan_azs mock."""
    with (
        patch("aws_bootstrap.cli.boto3.Session") as session,
        patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-1", "Name": "DLAMI"}),
        patch("aws_bootstrap.cli.instance_type_architecture", return_value="x86_64"),
        patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key"),
        patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-1"),
        patch("aws_bootstrap.cli.plan_azs") as plan,
    ):
        session.return_value.region_name = None
        plan.side_effect = lambda ec2, region, *a, **kw: {"us-west-2": PLAN_USW2, "us-east-1": PLAN_USE1}[region]
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
)
def test_launch_rejects_bad_az_flags_before_any_aws_call(args, message, key):
    with patch("aws_bootstrap.cli.boto3.Session") as session:
        result = CliRunner().invoke(main, ["launch", "--key-path", str(key), "--dry-run", *args])
    assert result.exit_code != 0
    assert message in result.output
    session.assert_not_called()


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_az_implies_its_region_and_is_planned_explicitly(dry_run_aws, key):
    args = ["-o", "json", "launch", "--key-path", str(key), "--dry-run", "--az", "us-west-2b"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["regions"] == ["us-west-2"]
    _, region, instance_type = dry_run_aws.call_args.args
    assert (region, instance_type) == ("us-west-2", "g4dn.xlarge")
    assert dry_run_aws.call_args.kwargs == {"explicit": ("us-west-2b",), "cheapest": False}


def test_dry_run_plans_every_region_not_only_the_first(dry_run_aws, key):
    regions = ["--region", "us-west-2", "--region", "us-east-1"]
    args = ["launch", "--key-path", str(key), "--dry-run", "--cheapest-az", *regions]
    result = CliRunner().invoke(main, ["-o", "json", *args])
    assert result.exit_code == 0, result.output
    zones = json.loads(result.output)["availability_zones"]
    assert zones == {
        "us-west-2": [{"zone": "us-west-2b", "spot_price": 0.9911}, {"zone": "us-west-2a", "spot_price": 2.2122}],
        "us-east-1": [{"zone": "us-east-1c", "spot_price": 2.0426}, {"zone": "us-east-1a", "spot_price": 2.0539}],
    }
    assert {c.args[1] for c in dry_run_aws.call_args_list} == {"us-west-2", "us-east-1"}


@pytest.mark.parametrize("fmt", ["text", "json", "yaml", "table"])
def test_dry_run_az_plan_renders_in_every_output_format(dry_run_aws, key, fmt):
    args = ["-o", fmt, "launch", "--key-path", str(key), "--dry-run", "--cheapest-az", "--region", "us-west-2"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    out = result.output
    if fmt == "json":
        assert json.loads(out)["availability_zones"]["us-west-2"][0]["zone"] == "us-west-2b"
    elif fmt == "yaml":
        assert yaml.safe_load(out)["availability_zones"]["us-west-2"][0]["zone"] == "us-west-2b"
    elif fmt == "table":
        assert "us-west-2: us-west-2b $0.991/h → us-west-2a $2.212/h" in out
        assert "{" not in out  # no Python repr of a nested mapping
    else:
        assert "us-west-2b $0.991/h → us-west-2a $2.212/h" in out


def test_dry_run_without_az_flags_reports_null(dry_run_aws, key):
    result = CliRunner().invoke(main, ["-o", "json", "launch", "--key-path", str(key), "--dry-run"])
    assert json.loads(result.output)["availability_zones"] is None
    dry_run_aws.assert_not_called()


# ---------------------------------------------------------------------------
# real launch: AZ plan threaded into the context, chosen AZ + price reported
# ---------------------------------------------------------------------------


def _launched(az: str, az_prices: dict, region: str = "us-west-2") -> RegionLaunch:
    ctx = RegionContext(
        region=region,
        ec2_client=MagicMock(),
        ami={"ImageId": "ami-1", "Name": "DLAMI"},
        sg_id="sg-1",
        key_name="k",
        az_prices=az_prices,
    )
    return RegionLaunch(region, ctx, {"InstanceId": "i-1", "Placement": {"AvailabilityZone": az}}, "spot")


@pytest.fixture
def real_launch_aws():
    with (
        patch("aws_bootstrap.cli.launch_with_retry") as launch,
        patch("aws_bootstrap.cli.wait_instance_ready") as ready,
        patch("aws_bootstrap.cli.wait_for_ssh", return_value=True),
        patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-gpu1"),
        patch("aws_bootstrap.cli.spot_price_or_none") as lookup,
    ):
        yield launch, ready, lookup


def test_launch_reports_the_chosen_az_and_its_spot_price(real_launch_aws, key):
    launch, ready, lookup = real_launch_aws
    launch.return_value = _launched("us-west-2b", {"us-west-2b": 0.9911})
    ready.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2b"}}
    result = CliRunner().invoke(main, ["-o", "json", "launch", "--key-path", str(key), "--no-setup"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["availability_zone"] == "us-west-2b"
    assert data["spot_price_per_hour"] == 0.9911
    lookup.assert_not_called()  # known from the AZ plan


def test_launch_looks_up_the_price_when_ec2_chose_the_az(real_launch_aws, key):
    launch, ready, lookup = real_launch_aws
    launch.return_value = _launched("us-west-2c", {})
    ready.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2c"}}
    lookup.return_value = 2.2122
    result = CliRunner().invoke(main, ["launch", "--key-path", str(key), "--no-setup"])
    assert result.exit_code == 0, result.output
    assert "Availability zone: us-west-2c (spot $2.212/h)" in result.output


def test_launch_price_lookup_failure_never_aborts(real_launch_aws, key):
    """The instance already exists: an unknown price is omitted, the record still emitted."""
    launch, ready, lookup = real_launch_aws
    launch.return_value = _launched("us-west-2c", {})
    ready.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-west-2c"}}
    lookup.return_value = None
    result = CliRunner().invoke(main, ["-o", "json", "launch", "--key-path", str(key), "--no-setup"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ready"
    assert "spot_price_per_hour" not in data


def test_prepare_region_threads_the_plan_into_the_context(key):
    captured: dict = {}

    def fake_launch(config, prepare_region, **kw):
        captured["ctx"] = prepare_region("us-west-2")
        raise CLIError("stop here")

    with (
        patch("aws_bootstrap.cli.boto3.Session"),
        patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-1", "Name": "DLAMI"}),
        patch("aws_bootstrap.cli.instance_type_architecture", return_value="x86_64"),
        patch("aws_bootstrap.cli.import_key_pair", return_value="k"),
        patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-1"),
        patch("aws_bootstrap.cli.plan_azs", return_value=PLAN_USW2),
        patch("aws_bootstrap.cli.launch_with_retry", side_effect=fake_launch),
    ):
        CliRunner().invoke(main, ["launch", "--key-path", str(key), "--cheapest-az", "--region", "us-west-2"])
    ctx = captured["ctx"]
    assert ctx.placement_azs == ("us-west-2b", "us-west-2a")
    assert ctx.az_prices == {"us-west-2b": 0.9911, "us-west-2a": 2.2122}


# ---------------------------------------------------------------------------
# pre-launch key check wiring
# ---------------------------------------------------------------------------


@pytest.mark.real_key_check
@pytest.mark.parametrize("dry_run", [True, False])
def test_unusable_key_fails_before_any_aws_call(key, dry_run):
    """key is a .pub with no private key next to it: the real check must stop the launch first."""
    with (
        patch("aws_bootstrap.cli.boto3.Session") as session,
        patch("aws_bootstrap.cli.launch_with_retry") as launch,
    ):
        args = ["launch", "--key-path", str(key), *(["--dry-run"] if dry_run else [])]
        result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    # A Session may be built to read the profile's region (local config); no AWS client is.
    session.return_value.client.assert_not_called()
    launch.assert_not_called()


def test_skip_key_check_bypasses_it(_stub_key_check, dry_run_aws, key):
    result = CliRunner().invoke(main, ["launch", "--key-path", str(key), "--dry-run", "--skip-key-check"])
    assert result.exit_code == 0, result.output
    _stub_key_check.assert_not_called()


def test_key_check_runs_on_dry_run(_stub_key_check, dry_run_aws, key):
    CliRunner().invoke(main, ["launch", "--key-path", str(key), "--dry-run"])
    _stub_key_check.assert_called_once_with(key)


# ---------------------------------------------------------------------------
# cluster launch
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_aws():
    with (
        patch("aws_bootstrap.cli.boto3.Session") as session,
        patch("aws_bootstrap.cli.find_cluster_instances") as existing,
        patch("aws_bootstrap.cli.ensure_cluster_placement_group") as pg,
        patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-1"),
        patch("aws_bootstrap.cli.ensure_cluster_security_group_rule"),
        patch("aws_bootstrap.cli.plan_azs", return_value=PLAN_USW2) as plan,
        patch("aws_bootstrap.cli.cluster_mod.launch_cluster_nodes") as nodes,
    ):
        session.return_value.region_name = None
        existing.return_value = []
        yield {"existing": existing, "pg": pg, "plan": plan, "nodes": nodes}


def test_cluster_cheapest_az_plans_the_first_node(cluster_aws, key):
    captured: dict = {}

    def fake_nodes(config, prepare_region, *a, **kw):
        with (
            patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-1"}),
            patch("aws_bootstrap.cli.instance_type_architecture", return_value="x86_64"),
            patch("aws_bootstrap.cli.import_key_pair", return_value="k"),
        ):
            captured["ctx"] = prepare_region("us-west-2")

    cluster_aws["nodes"].side_effect = fake_nodes
    args = ["cluster", "launch", "--cluster-id", "c1", "--key-path", str(key), "--no-setup", "--cheapest-az"]
    result = CliRunner().invoke(main, [*args, "--region", "us-west-2"])
    assert result.exit_code == 0, result.output
    assert cluster_aws["plan"].call_args.kwargs == {"explicit": (), "cheapest": True}
    assert captured["ctx"].placement_azs == ("us-west-2b", "us-west-2a")


def test_cluster_az_implies_region(cluster_aws, key):
    args = ["cluster", "launch", "--cluster-id", "c1", "--key-path", str(key), "--no-setup", "--az", "us-east-1c"]
    CliRunner().invoke(main, args)
    assert cluster_aws["plan"].call_args.args[1] == "us-east-1"


def test_cluster_existing_az_conflict_is_an_error(cluster_aws, key):
    cluster_aws["existing"].return_value = [{"AvailabilityZone": "us-west-2a", "Rank": 0, "InstanceId": "i-0"}]
    args = ["cluster", "launch", "--cluster-id", "c1", "--nodes", "3", "--key-path", str(key), "--az", "us-west-2b"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    assert "is in us-west-2a" in result.output
    cluster_aws["nodes"].assert_not_called()


@pytest.mark.real_key_check
def test_cluster_unusable_key_fails_before_creating_anything(cluster_aws, key):
    result = CliRunner().invoke(main, ["cluster", "launch", "--cluster-id", "c1", "--key-path", str(key)])
    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    cluster_aws["pg"].assert_not_called()
    cluster_aws["nodes"].assert_not_called()
