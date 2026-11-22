"""What `launch --dry-run` reports for choosing a pool, and the on-demand price after a launch.

On-demand price, spot pools with placement scores, and quota headroom are display only:
lookups are stubbed by the ``launch_aws`` fixture (``launch_aws.pools``) and set per test.
"""

from __future__ import annotations
import json
from unittest.mock import patch

import botocore.exceptions
import pytest

from aws_bootstrap.cli import main
from aws_bootstrap.ec2 import AzOption, CLIError


G7E = ["--instance-type", "g7e.2xlarge", "--region", "us-east-1"]
POOLS = [AzOption("us-east-1a", 2.1339), AzOption("us-east-1c", 2.2767)]
SCORES = {"us-east-1": {"us-east-1a": 1, "us-east-1c": 9}}
FIT = {  # the spot pool is full; on-demand has room
    "family": "gvt",
    "spot": {"quota": 8, "in_use": 8, "needed": 8, "fits": False},
    "on-demand": {"quota": 8, "in_use": 0, "needed": 8, "fits": True},
}
RAISE_SPOT = "aws-bootstrap quota request --family gvt --type spot --desired-value 16 --region us-east-1"


@pytest.fixture
def pools(launch_aws):
    """Known facts for g7e.2xlarge in us-east-1: on-demand $3.363/h, two spot pools, a full spot quota."""
    launch_aws.pools.od_price.return_value = 3.36312
    launch_aws.pools.scores.return_value = SCORES
    launch_aws.pools.quota_fit.return_value = FIT
    with patch("aws_bootstrap.cli.plan_azs", return_value=POOLS) as plan:
        launch_aws.pools.plan = plan
        yield launch_aws.pools


def _dry_run(runner, key_path, *args: str, fmt: str = "text"):
    return runner.invoke(main, ["-o", fmt, "launch", "--key-path", str(key_path), "--dry-run", *G7E, *args])


@pytest.mark.parametrize(
    "args,present,absent",
    [
        pytest.param(
            [],
            [
                "On-demand price: $3.363/h",
                "Spot pools: us-east-1a $2.134/h score 1 · us-east-1c $2.277/h score 9",
                "Quota (gvt): spot 8/8 vCPU in use + 8 → won't fit; on-demand 0/8 vCPU in use + 8 → fits",
                RAISE_SPOT,
            ],
            [],
            id="spot",
        ),
        pytest.param(
            ["--on-demand"],
            ["On-demand price: $3.363/h", "on-demand 0/8 vCPU in use + 8 → fits"],
            ["Spot pools", "quota request"],  # on-demand fits, and spot pools don't matter
            id="on-demand",
        ),
    ],
)
def test_dry_run_text_shows_what_decides_the_pool(runner, pools, key_path, args, present, absent):
    result = _dry_run(runner, key_path, *args)
    assert result.exit_code == 0, result.output
    for text in present:
        assert text in result.output
    for text in absent:
        assert text not in result.output


def test_dry_run_structured_output_carries_the_pool_facts(runner, pools, key_path, structured):
    fmt, load = structured
    result = _dry_run(runner, key_path, fmt=fmt)
    assert result.exit_code == 0, result.output
    out = load(result.output)
    assert out["on_demand_price_per_hour"] == {"us-east-1": 3.36312}
    assert out["spot_pools"] == {
        "us-east-1": [
            {"zone": "us-east-1a", "spot_price": 2.1339, "placement_score": 1},
            {"zone": "us-east-1c", "spot_price": 2.2767, "placement_score": 9},
        ]
    }
    assert out["quota_fit"] == {"us-east-1": FIT}


def test_dry_run_table_flattens_the_pool_facts(runner, pools, key_path):
    result = _dry_run(runner, key_path, fmt="table")
    assert result.exit_code == 0, result.output
    for text in ("us-east-1: $3.363/h", "us-east-1c $2.277/h score 9", "spot 8/8 vCPU in use + 8 → won't fit"):
        assert text in result.output


def test_placement_scores_join_the_az_plan_without_reordering_it(runner, pools, key_path):
    result = _dry_run(runner, key_path, "--cheapest-az", fmt="json")
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)["availability_zones"]["us-east-1"]
    assert [(o["zone"], o["placement_score"]) for o in plan] == [("us-east-1a", 1), ("us-east-1c", 9)]  # price order


def test_a_cheapest_az_plan_doubles_as_the_spot_pools(runner, pools, key_path):
    """It already lists every usable AZ: no second lookup, and no second, near-identical line."""
    result = _dry_run(runner, key_path, "--cheapest-az")
    assert result.exit_code == 0, result.output
    assert "AZs: us-east-1a $2.134/h score 1 → us-east-1c $2.277/h score 9" in result.output
    assert "Spot pools" not in result.output
    pools.plan.assert_called_once()


@pytest.mark.parametrize(
    "error,pools_line",
    [
        pytest.param(
            CLIError("g7e.2xlarge is not offered in us-east-1."),
            "Spot pools: g7e.2xlarge is not offered in us-east-1.",
            id="not-offered",
        ),
        pytest.param(
            botocore.exceptions.EndpointConnectionError(endpoint_url="https://ec2"),
            "Spot pools: unknown (Could not connect",
            id="unreachable",
        ),
    ],
)
def test_dry_run_reports_unknown_facts_without_failing(runner, launch_aws, key_path, error, pools_line):
    with patch("aws_bootstrap.cli.plan_azs", side_effect=error):
        result = _dry_run(runner, key_path)
    assert result.exit_code == 0, result.output
    for text in ("On-demand price: unknown", pools_line, "Quota (gvt): unknown"):
        assert text in result.output


def test_dry_run_says_a_type_has_no_gpu_quota(runner, launch_aws, key_path):
    args = ["--key-path", str(key_path), "--dry-run", "--instance-type", "t3.medium", "--region", "us-east-1"]
    result = runner.invoke(main, ["launch", *args])
    assert result.exit_code == 0, result.output
    assert "Quota: n/a (no GPU vCPU quota for this type)" in result.output
    launch_aws.pools.quota_fit.assert_not_called()


def test_dry_run_labels_each_region(runner, pools, key_path):
    result = _dry_run(runner, key_path, "--region", "us-west-2")
    assert result.exit_code == 0, result.output
    for region in ("us-east-1", "us-west-2"):
        assert f"On-demand price ({region}): $3.363/h" in result.output
        assert f"Quota ({region}, gvt):" in result.output
    pools.scores.assert_called_once()  # one placement-score call covers every region
    assert pools.scores.call_args.args[2] == ("us-east-1", "us-west-2")


def test_each_regions_pools_get_that_regions_scores(runner, pools, key_path):
    result = _dry_run(runner, key_path, "--region", "us-west-2", fmt="json")  # SCORES only has us-east-1
    spot_pools = json.loads(result.output)["spot_pools"]
    assert [o["placement_score"] for o in spot_pools["us-east-1"]] == [1, 9]
    assert [o["placement_score"] for o in spot_pools["us-west-2"]] == [None, None]


@pytest.mark.parametrize("price", [3.36312, None], ids=["known", "unknown"])
def test_launch_reports_the_on_demand_price_when_known(runner, launched, key_path, price):
    launched.pools.od_price.return_value = price
    result = runner.invoke(main, ["-o", "json", "launch", "--key-path", str(key_path), "--no-setup"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output).get("on_demand_price_per_hour") == price


def test_launch_text_shows_the_on_demand_price_once_set_up(runner, launched, key_path):
    launched.pools.od_price.return_value = 3.36312
    result = runner.invoke(main, ["launch", "--key-path", str(key_path), "--no-setup"])
    assert result.exit_code == 0, result.output
    assert "On-demand price: $3.363/h" in result.output
    launched.pools.od_price.assert_called_once()  # looked up once, after the alias was written
