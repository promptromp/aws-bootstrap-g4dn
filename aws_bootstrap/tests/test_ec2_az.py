"""AZ selection: spot-price ranking (plan_azs and its helpers) and the AZ-aware launch sweep."""

from __future__ import annotations
from datetime import UTC, datetime
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from aws_bootstrap.config import LaunchConfig
from aws_bootstrap.ec2 import (
    AzOption,
    AzUnsupportedError,
    CLIError,
    RegionContext,
    _run_instances,
    default_subnet_azs,
    launch_with_retry,
    offered_azs,
    plan_azs,
    spot_price_or_none,
    spot_prices_by_az,
)


def _client_error(code: str, message: str = "test") -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError({"Error": {"Code": code, "Message": message}}, "Op")


CAPACITY = _client_error("InsufficientInstanceCapacity")
QUOTA = _client_error("MaxSpotInstanceCountExceeded")
PRICE = _client_error("SpotMaxPriceTooLow")
UNSUPPORTED = _client_error("Unsupported", "Your requested instance type is not supported in your requested AZ.")


# ---------------------------------------------------------------------------
# helpers: spot prices, offerings, default subnets
# ---------------------------------------------------------------------------


def _ec2(*, prices=None, offered=None, subnets=None, zones_error=None, opt_in=None, price_error=None) -> MagicMock:
    """A fake EC2 client whose paginators return the given pages.

    prices: [(az, price, hour)]; offered / subnets: [az]; opt_in: {az: status}.
    """
    ec2 = MagicMock()
    pages = {
        "describe_spot_price_history": [
            {
                "SpotPriceHistory": [
                    {"AvailabilityZone": az, "SpotPrice": str(p), "Timestamp": datetime(2026, 11, 21, h, tzinfo=UTC)}
                    for az, p, h in (prices or [])
                ]
            }
        ],
        "describe_instance_type_offerings": [{"InstanceTypeOfferings": [{"Location": z} for z in (offered or [])]}],
        "describe_subnets": [{"Subnets": [{"AvailabilityZone": z} for z in (subnets or [])]}],
    }

    def paginator(name):
        pag = MagicMock()
        if name == "describe_spot_price_history" and price_error:
            pag.paginate.side_effect = price_error
        else:
            pag.paginate.return_value = pages[name]
        return pag

    ec2.get_paginator.side_effect = paginator
    if zones_error:
        ec2.describe_availability_zones.side_effect = zones_error
    else:
        ec2.describe_availability_zones.side_effect = lambda ZoneNames, **kw: {
            "AvailabilityZones": [
                {"ZoneName": z, "OptInStatus": (opt_in or {}).get(z, "opt-in-not-required")} for z in ZoneNames
            ]
        }
    return ec2


USW2 = ["us-west-2a", "us-west-2b", "us-west-2c", "us-west-2d"]
USW2_PRICES = [("us-west-2a", 2.2122, 13), ("us-west-2b", 0.9911, 19), ("us-west-2c", 2.2122, 16)]


def test_spot_prices_by_az_keeps_the_latest_record_per_az():
    ec2 = _ec2(prices=[("us-west-2a", 2.5, 10), ("us-west-2a", 2.2, 13), ("us-west-2b", 0.99, 19)])
    assert spot_prices_by_az(ec2, "g6e.2xlarge") == {"us-west-2a": 2.2, "us-west-2b": 0.99}


def test_offered_and_default_subnet_azs():
    ec2 = _ec2(offered=USW2, subnets=["us-west-2a", "us-west-2b"])
    assert offered_azs(ec2, "g6e.2xlarge") == set(USW2)
    assert default_subnet_azs(ec2) == {"us-west-2a", "us-west-2b"}


@pytest.mark.parametrize(
    "side_effect",
    [_client_error("UnauthorizedOperation"), botocore.exceptions.EndpointConnectionError(endpoint_url="x")],
    ids=["client-error", "botocore-error"],
)
def test_spot_price_or_none_swallows_api_errors(side_effect):
    """Called after RunInstances: a denied/throttled lookup must not abort and lose the instance record."""
    ec2 = MagicMock()
    ec2.describe_spot_price_history.side_effect = side_effect
    assert spot_price_or_none(ec2, "g6e.2xlarge", "us-west-2b") is None


def test_spot_price_or_none_returns_the_price():
    ec2 = MagicMock()
    ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": [{"SpotPrice": "0.9911"}]}
    assert spot_price_or_none(ec2, "g6e.2xlarge", "us-west-2b") == pytest.approx(0.9911)


# ---------------------------------------------------------------------------
# plan_azs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan_kwargs,subnets,expected",
    [
        ({"cheapest": True}, USW2, ["us-west-2b", "us-west-2a", "us-west-2c", "us-west-2d"]),  # 2d: no price, last
        ({"explicit": ("us-west-2c", "us-west-2b")}, USW2, ["us-west-2c", "us-west-2b"]),  # given order kept
        ({"explicit": ("us-west-2c", "us-west-2b"), "cheapest": True}, USW2, ["us-west-2b", "us-west-2c"]),
        ({"cheapest": True}, ["us-west-2a", "us-west-2c"], ["us-west-2a", "us-west-2c"]),  # default VPC subnets only
        ({"cheapest": True}, [], ["us-west-2b", "us-west-2a", "us-west-2c", "us-west-2d"]),  # no default subnets
    ],
    ids=["cheapest", "explicit-order", "explicit-cheapest", "default-subnet-filter", "no-default-subnets"],
)
def test_plan_azs(plan_kwargs, subnets, expected):
    plan = plan_azs(_ec2(prices=USW2_PRICES, offered=USW2, subnets=subnets), "us-west-2", "g6e.2xlarge", **plan_kwargs)
    assert [o.zone for o in plan] == expected
    prices = {az: p for az, p, _ in USW2_PRICES}
    assert all(o == AzOption(o.zone, prices.get(o.zone)) for o in plan)


@pytest.mark.parametrize(
    "ec2_kwargs,plan_kwargs,match",
    [
        (
            {"offered": ["us-west-2a", "us-west-2b"]},
            {"explicit": ("us-west-2d",)},
            "Usable AZs in us-west-2: us-west-2a",
        ),
        (
            {"offered": USW2, "zones_error": _client_error("InvalidParameterValue", "Invalid AZ")},
            {"explicit": ("us-west-2z",)},
            "Unknown availability zone in us-west-2: us-west-2z",
        ),
        (
            {"offered": USW2, "opt_in": {"us-west-2-lax-1a": "not-opted-in"}},
            {"explicit": ("us-west-2-lax-1a",)},
            "not opted in",
        ),
        ({"offered": []}, {"cheapest": True}, "not offered in any availability zone of us-west-2"),
    ],
    ids=["unusable-explicit", "unknown-az", "not-opted-in", "type-not-offered"],
)
def test_plan_azs_errors(ec2_kwargs, plan_kwargs, match):
    with pytest.raises(CLIError, match=match):
        plan_azs(_ec2(prices=USW2_PRICES, subnets=USW2, **ec2_kwargs), "us-west-2", "g6e.2xlarge", **plan_kwargs)


def test_plan_azs_explicit_tolerates_a_denied_price_lookup():
    """--az without --cheapest-az only shows prices; it mustn't need DescribeSpotPriceHistory."""
    ec2 = _ec2(offered=USW2, subnets=USW2, price_error=_client_error("UnauthorizedOperation"))
    assert plan_azs(ec2, "us-west-2", "g6e.2xlarge", explicit=("us-west-2b",)) == [AzOption("us-west-2b", None)]
    with pytest.raises(botocore.exceptions.ClientError):
        plan_azs(ec2, "us-west-2", "g6e.2xlarge", cheapest=True)  # the ranking needs prices


# ---------------------------------------------------------------------------
# _run_instances: Unsupported in a pinned AZ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "placement_az,raised", [("us-west-2d", AzUnsupportedError), (None, botocore.exceptions.ClientError)]
)
def test_run_instances_unsupported(placement_az, raised):
    ec2 = MagicMock()
    ec2.run_instances.side_effect = UNSUPPORTED
    with pytest.raises(raised) as exc:
        _run_instances(ec2, LaunchConfig(), "ami", "sg", "us-west-2", True, "k", placement_az=placement_az)
    assert "not supported in your requested AZ" in str(exc.value)  # AWS's own message is kept


# ---------------------------------------------------------------------------
# launch_with_retry: the AZ loop inside the region sweep
# ---------------------------------------------------------------------------


def _ctx(region: str, azs: tuple[str, ...], outcomes: dict[str | None, object]) -> RegionContext:
    """Region context whose run_instances outcome depends on the requested AZ (default: no capacity)."""
    client = MagicMock()

    def run(**params):
        az = params.get("Placement", {}).get("AvailabilityZone")
        outcome = outcomes.get(az, CAPACITY)
        if isinstance(outcome, Exception):
            raise outcome
        return {"Instances": [{"InstanceId": outcome, "Placement": {"AvailabilityZone": az}}]}

    client.run_instances.side_effect = run
    return RegionContext(
        region=region, ec2_client=client, ami={"ImageId": "ami-1"}, sg_id="sg-1", key_name="k", placement_azs=azs
    )


W2 = ("us-west-2b", "us-west-2a")


@pytest.mark.parametrize(
    "w2_outcomes,expected_region,expected_attempts,expected_fatal",
    [
        ({"us-west-2a": "i-a"}, "us-west-2", ["us-west-2b", "us-west-2a"], []),
        ({}, "us-east-1", ["us-west-2b", "us-west-2a", None], []),
        ({"us-west-2b": QUOTA, "us-west-2a": "i-a"}, "us-east-1", ["us-west-2b", None], [("us-west-2", "quota")]),
        ({"us-west-2b": PRICE, "us-west-2a": "i-a"}, "us-west-2", ["us-west-2b", "us-west-2a"], []),
        ({"us-west-2b": UNSUPPORTED, "us-west-2a": "i-a"}, "us-west-2", ["us-west-2b", "us-west-2a"], []),
        (
            {"us-west-2b": PRICE, "us-west-2a": UNSUPPORTED},
            "us-east-1",
            ["us-west-2b", "us-west-2a", None],
            [("us-west-2", "price")],
        ),
        (
            {"us-west-2b": UNSUPPORTED, "us-west-2a": UNSUPPORTED},
            "us-east-1",
            ["us-west-2b", "us-west-2a", None],
            [("us-west-2", "unsupported")],
        ),
    ],
    ids=[
        "capacity-tries-next-az",
        "all-azs-miss-next-region",
        "quota-skips-other-azs",
        "price-tries-next-az",
        "unsupported-tries-next-az",
        "price-and-unsupported-drop-region",
        "all-unsupported-drop-region",
    ],
)
def test_az_sweep(w2_outcomes, expected_region, expected_attempts, expected_fatal):
    contexts = {"us-west-2": _ctx("us-west-2", W2, w2_outcomes), "us-east-1": _ctx("us-east-1", (), {None: "i-east"})}
    attempts: list[str | None] = []
    fatal: list[tuple[str, str]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1")),
        lambda r: contexts[r],
        on_attempt=lambda region, market, attempt, az: attempts.append(az),
        on_region_fatal=lambda region, kind, msg: fatal.append((region, kind)),
    )
    assert result.region == expected_region
    assert attempts == expected_attempts
    assert fatal == expected_fatal


def test_hard_pinned_az_keeps_the_pre_az_selection_behaviour():
    """One AZ (an EBS volume's, a cluster's, a single --az): Unsupported surfaces AWS's own error."""
    contexts = {"us-west-2": _ctx("us-west-2", ("us-west-2d",), {"us-west-2d": UNSUPPORTED})}
    with pytest.raises(botocore.exceptions.ClientError) as exc:
        launch_with_retry(LaunchConfig(regions=("us-west-2",)), lambda r: contexts[r])
    assert exc.value.response["Error"]["Code"] == "Unsupported"


def test_hard_pinned_az_price_failure_drops_the_region():
    contexts = {"us-west-2": _ctx("us-west-2", ("us-west-2d",), {"us-west-2d": PRICE})}
    fatal: list[str] = []
    with pytest.raises(CLIError):
        launch_with_retry(
            LaunchConfig(regions=("us-west-2",), spot=False),
            lambda r: contexts[r],
            on_region_fatal=lambda region, kind, msg: fatal.append(kind),
        )
    assert fatal == ["price"]


def test_all_unsupported_skips_the_on_demand_fallback():
    """On-demand would hit the same unsupported AZs: fail with AWS's reason, don't offer it."""
    contexts = {"us-west-2": _ctx("us-west-2", W2, {"us-west-2b": UNSUPPORTED, "us-west-2a": UNSUPPORTED})}
    confirm = MagicMock(return_value=True)
    with pytest.raises(CLIError) as exc:
        launch_with_retry(LaunchConfig(regions=("us-west-2",)), lambda r: contexts[r], confirm_on_demand=confirm)
    confirm.assert_not_called()
    msg = exc.value.format_message()
    assert "Could not launch g4dn.xlarge (spot)" in msg
    assert "not supported in your requested AZ" in msg
    assert "list instance-types --prefix g4dn" in msg


def test_failure_names_only_the_azs_actually_tried():
    contexts = {"us-west-2": _ctx("us-west-2", ("us-west-2b", "us-west-2a", "us-west-2c"), {"us-west-2b": QUOTA})}
    with pytest.raises(CLIError) as exc:
        launch_with_retry(
            LaunchConfig(regions=("us-west-2",), spot=False), lambda r: contexts[r], confirm_on_demand=lambda: False
        )
    assert "(tried us-west-2b)" in exc.value.format_message()


def test_wait_resweeps_the_azs_and_the_final_error_names_them():
    contexts = {"us-west-2": _ctx("us-west-2", W2, {})}
    clock = iter([0.0, 10.0, 10.0, 99999.0, 99999.0])
    attempts: list[str | None] = []
    with pytest.raises(CLIError) as exc:
        launch_with_retry(
            LaunchConfig(regions=("us-west-2",), wait=True, wait_timeout=60),
            lambda r: contexts[r],
            on_attempt=lambda region, market, attempt, az: attempts.append(az),
            sleeper=lambda s: None,
            clock=lambda: next(clock),
        )
    assert attempts[:4] == ["us-west-2b", "us-west-2a", "us-west-2b", "us-west-2a"]
    assert "us-west-2: no spot capacity (tried us-west-2b, us-west-2a)" in exc.value.format_message()
