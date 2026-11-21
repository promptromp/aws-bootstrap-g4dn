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


def _ts(hour: int) -> datetime:
    return datetime(2026, 11, 21, hour, tzinfo=UTC)


def _ec2(*, prices=None, offered=None, subnets=None, zones_error=None, opt_in=None) -> MagicMock:
    """A fake EC2 client whose paginators return the given pages.

    prices: [(az, price, hour)]; offered / subnets: [az]; opt_in: {az: status}.
    """
    ec2 = MagicMock()
    pages = {
        "describe_spot_price_history": [
            {
                "SpotPriceHistory": [
                    {"AvailabilityZone": az, "SpotPrice": str(p), "Timestamp": _ts(h)} for az, p, h in (prices or [])
                ]
            }
        ],
        "describe_instance_type_offerings": [{"InstanceTypeOfferings": [{"Location": z} for z in (offered or [])]}],
        "describe_subnets": [{"Subnets": [{"AvailabilityZone": z} for z in (subnets or [])]}],
    }

    def paginator(name):
        pag = MagicMock()
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_spot_prices_by_az_keeps_the_latest_record_per_az():
    ec2 = _ec2(prices=[("us-west-2a", 2.5, 10), ("us-west-2a", 2.2, 13), ("us-west-2b", 0.99, 19)])
    assert spot_prices_by_az(ec2, "g6e.2xlarge") == {"us-west-2a": 2.2, "us-west-2b": 0.99}


def test_offered_and_default_subnet_azs():
    ec2 = _ec2(offered=USW2, subnets=["us-west-2a", "us-west-2b"])
    assert offered_azs(ec2, "g6e.2xlarge") == set(USW2)
    assert default_subnet_azs(ec2) == {"us-west-2a", "us-west-2b"}


def test_spot_price_or_none_swallows_api_errors():
    """Called after RunInstances: a denied/throttled lookup must not abort and lose the instance record."""
    ec2 = MagicMock()
    ec2.describe_spot_price_history.side_effect = _client_error("UnauthorizedOperation")
    assert spot_price_or_none(ec2, "g6e.2xlarge", "us-west-2b") is None
    ec2.describe_spot_price_history.side_effect = botocore.exceptions.EndpointConnectionError(endpoint_url="x")
    assert spot_price_or_none(ec2, "g6e.2xlarge", "us-west-2b") is None
    ec2.describe_spot_price_history.side_effect = None
    ec2.describe_spot_price_history.return_value = {"SpotPriceHistory": [{"SpotPrice": "0.9911"}]}
    assert spot_price_or_none(ec2, "g6e.2xlarge", "us-west-2b") == pytest.approx(0.9911)


# ---------------------------------------------------------------------------
# plan_azs
# ---------------------------------------------------------------------------


def test_plan_azs_cheapest_orders_by_price_unknown_last():
    ec2 = _ec2(prices=USW2_PRICES, offered=USW2, subnets=USW2)
    plan = plan_azs(ec2, "us-west-2", "g6e.2xlarge", cheapest=True)
    assert [o.zone for o in plan] == ["us-west-2b", "us-west-2a", "us-west-2c", "us-west-2d"]
    assert plan[0] == AzOption("us-west-2b", pytest.approx(0.9911))
    assert plan[-1].spot_price is None  # us-west-2d: offered, no price record


def test_plan_azs_explicit_keeps_the_given_order_with_prices():
    ec2 = _ec2(prices=USW2_PRICES, offered=USW2, subnets=USW2)
    plan = plan_azs(ec2, "us-west-2", "g6e.2xlarge", explicit=("us-west-2c", "us-west-2b"))
    assert [(o.zone, o.spot_price) for o in plan] == [("us-west-2c", 2.2122), ("us-west-2b", 0.9911)]


def test_plan_azs_explicit_with_cheapest_ranks_only_the_given_azs():
    ec2 = _ec2(prices=USW2_PRICES, offered=USW2, subnets=USW2)
    plan = plan_azs(ec2, "us-west-2", "g6e.2xlarge", explicit=("us-west-2c", "us-west-2b"), cheapest=True)
    assert [o.zone for o in plan] == ["us-west-2b", "us-west-2c"]


def test_plan_azs_limits_candidates_to_azs_with_a_default_subnet():
    """The tool launches into the default VPC, so an AZ without a default subnet can't be used."""
    ec2 = _ec2(prices=USW2_PRICES, offered=USW2, subnets=["us-west-2a", "us-west-2c"])
    assert [o.zone for o in plan_azs(ec2, "us-west-2", "g6e.2xlarge", cheapest=True)] == ["us-west-2a", "us-west-2c"]


def test_plan_azs_no_default_subnets_at_all_does_not_filter():
    ec2 = _ec2(prices=USW2_PRICES, offered=USW2, subnets=[])
    assert len(plan_azs(ec2, "us-west-2", "g6e.2xlarge", cheapest=True)) == 4


def test_plan_azs_explicit_unusable_az_names_the_usable_ones():
    ec2 = _ec2(prices=USW2_PRICES, offered=["us-west-2a", "us-west-2b"], subnets=USW2)
    with pytest.raises(CLIError) as exc:
        plan_azs(ec2, "us-west-2", "g6e.2xlarge", explicit=("us-west-2d",))
    msg = exc.value.format_message()
    assert "us-west-2d" in msg
    assert "Usable AZs in us-west-2: us-west-2a, us-west-2b" in msg


def test_plan_azs_unknown_az_is_confirmed_with_ec2():
    ec2 = _ec2(offered=USW2, subnets=USW2, zones_error=_client_error("InvalidParameterValue", "Invalid AZ"))
    with pytest.raises(CLIError, match="Unknown availability zone in us-west-2: us-west-2z"):
        plan_azs(ec2, "us-west-2", "g6e.2xlarge", explicit=("us-west-2z",))


def test_plan_azs_rejects_a_zone_not_opted_in():
    ec2 = _ec2(offered=USW2, subnets=USW2, opt_in={"us-west-2-lax-1a": "not-opted-in"})
    with pytest.raises(CLIError, match="not opted in"):
        plan_azs(ec2, "us-west-2", "g6e.2xlarge", explicit=("us-west-2-lax-1a",))


def test_plan_azs_type_not_offered_in_region():
    with pytest.raises(CLIError, match="not offered in any availability zone of us-west-2"):
        plan_azs(_ec2(offered=[]), "us-west-2", "p6-b300.48xlarge", cheapest=True)


# ---------------------------------------------------------------------------
# _run_instances: Unsupported in a pinned AZ
# ---------------------------------------------------------------------------


def test_run_instances_unsupported_in_pinned_az_is_az_level():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _client_error("Unsupported")
    with pytest.raises(AzUnsupportedError, match="us-west-2d"):
        _run_instances(ec2, LaunchConfig(), "ami", "sg", "us-west-2", True, "k", placement_az="us-west-2d")


def test_run_instances_unsupported_without_az_is_unchanged():
    ec2 = MagicMock()
    ec2.run_instances.side_effect = _client_error("Unsupported")
    with pytest.raises(botocore.exceptions.ClientError):
        _run_instances(ec2, LaunchConfig(), "ami", "sg", "us-west-2", True, "k")


# ---------------------------------------------------------------------------
# launch_with_retry: AZ fallthrough
# ---------------------------------------------------------------------------


def _ctx(region: str, azs: tuple[str, ...], outcomes: dict[str | None, object]) -> RegionContext:
    """Region context whose run_instances outcome depends on the requested AZ."""
    client = MagicMock()

    def run(**params):
        az = params.get("Placement", {}).get("AvailabilityZone")
        outcome = outcomes.get(az, _client_error("InsufficientInstanceCapacity"))
        if isinstance(outcome, Exception):
            raise outcome
        return {"Instances": [{"InstanceId": outcome, "Placement": {"AvailabilityZone": az}}]}

    client.run_instances.side_effect = run
    return RegionContext(
        region=region,
        ec2_client=client,
        ami={"ImageId": "ami-1"},
        sg_id="sg-1",
        key_name="k",
        placement_azs=azs,
    )


def test_capacity_miss_tries_the_next_az_before_the_next_region():
    contexts = {
        "us-west-2": _ctx("us-west-2", ("us-west-2b", "us-west-2a"), {"us-west-2a": "i-a"}),
        "us-east-1": _ctx("us-east-1", (), {None: "i-east"}),
    }
    attempts: list[tuple[str, str | None]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1")),
        lambda r: contexts[r],
        on_attempt=lambda region, market, attempt, az: attempts.append((region, az)),
    )
    assert result.region == "us-west-2"
    assert result.instance["InstanceId"] == "i-a"
    assert attempts == [("us-west-2", "us-west-2b"), ("us-west-2", "us-west-2a")]


def test_every_az_missing_falls_through_to_the_next_region():
    contexts = {
        "us-west-2": _ctx("us-west-2", ("us-west-2b", "us-west-2a"), {}),
        "us-east-1": _ctx("us-east-1", (), {None: "i-east"}),
    }
    attempts: list[tuple[str, str | None]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1")),
        lambda r: contexts[r],
        on_attempt=lambda region, market, attempt, az: attempts.append((region, az)),
    )
    assert result.region == "us-east-1"
    assert attempts == [("us-west-2", "us-west-2b"), ("us-west-2", "us-west-2a"), ("us-east-1", None)]


def test_region_fatal_in_one_az_skips_the_regions_other_azs():
    """Quota / spot price is region-wide: trying the other AZs would only repeat it."""
    quota = _client_error("MaxSpotInstanceCountExceeded")
    contexts = {
        "us-west-2": _ctx("us-west-2", ("us-west-2b", "us-west-2a"), {"us-west-2b": quota, "us-west-2a": "i-a"}),
        "us-east-1": _ctx("us-east-1", (), {None: "i-east"}),
    }
    fatal: list[tuple[str, str]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1")),
        lambda r: contexts[r],
        on_region_fatal=lambda region, kind, msg: fatal.append((region, kind)),
    )
    assert result.region == "us-east-1"
    assert fatal == [("us-west-2", "quota")]
    assert contexts["us-west-2"].ec2_client.run_instances.call_count == 1


def test_unsupported_in_every_az_drops_the_region():
    unsupported = _client_error("Unsupported")
    contexts = {
        "us-west-2": _ctx("us-west-2", ("us-west-2d",), {"us-west-2d": unsupported}),
        "us-east-1": _ctx("us-east-1", (), {None: "i-east"}),
    }
    fatal: list[tuple[str, str]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1")),
        lambda r: contexts[r],
        on_region_fatal=lambda region, kind, msg: fatal.append((region, kind)),
    )
    assert result.region == "us-east-1"
    assert fatal == [("us-west-2", "unsupported")]


def test_wait_resweeps_the_azs_and_the_final_error_names_them():
    contexts = {"us-west-2": _ctx("us-west-2", ("us-west-2b", "us-west-2a"), {})}
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


def test_no_azs_lets_ec2_choose():
    contexts = {"us-west-2": _ctx("us-west-2", (), {None: "i-any"})}
    result = launch_with_retry(LaunchConfig(regions=("us-west-2",)), lambda r: contexts[r])
    assert result.instance["InstanceId"] == "i-any"
    params = contexts["us-west-2"].ec2_client.run_instances.call_args.kwargs
    assert "Placement" not in params
