"""AZ selection: spot-price ranking (plan_azs and its helpers) and the AZ-aware launch sweep."""

from __future__ import annotations
import json
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
    default_vcpus,
    instance_type_vcpus,
    launch_with_retry,
    offered_azs,
    on_demand_price,
    plan_azs,
    spot_placement_scores,
    spot_price_or_none,
    spot_prices_by_az,
)
from aws_bootstrap.tests.helpers import client_error, session_with_clients


CAPACITY = client_error("InsufficientInstanceCapacity")
QUOTA = client_error("MaxSpotInstanceCountExceeded")
PRICE = client_error("SpotMaxPriceTooLow")
UNSUPPORTED = client_error("Unsupported", "Your requested instance type is not supported in your requested AZ.")


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
    [client_error("UnauthorizedOperation"), botocore.exceptions.EndpointConnectionError(endpoint_url="x")],
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
            {"offered": USW2, "zones_error": client_error("InvalidParameterValue", "Invalid AZ")},
            {"explicit": ("us-west-2z",)},
            "Unknown availability zone in us-west-2: us-west-2z",
        ),
        (
            {"offered": USW2, "opt_in": {"us-west-2-lax-1a": "not-opted-in"}},
            {"explicit": ("us-west-2-lax-1a",)},
            "not opted in",
        ),
        ({"offered": []}, {"cheapest": True}, "not offered in any availability zone of us-west-2"),
        (
            {"offered": ["us-west-2d"], "subnets_override": ["us-west-2a"]},
            {"cheapest": True},
            "only in AZs without a default subnet",
        ),
        (
            {"offered": USW2, "zones_error": botocore.exceptions.EndpointConnectionError(endpoint_url="x")},
            {"explicit": ("us-west-2b",)},
            "Can't reach EC2 in region us-west-2",
        ),
    ],
    ids=[
        "unusable-explicit",
        "unknown-az",
        "not-opted-in",
        "type-not-offered",
        "no-default-subnet",
        "unreachable-region",
    ],
)
def test_plan_azs_errors(ec2_kwargs, plan_kwargs, match):
    with pytest.raises(CLIError, match=match):
        subnets = ec2_kwargs.pop("subnets_override", USW2)
        plan_azs(_ec2(prices=USW2_PRICES, subnets=subnets, **ec2_kwargs), "us-west-2", "g6e.2xlarge", **plan_kwargs)


def test_plan_azs_explicit_tolerates_a_denied_price_lookup():
    """--az without --cheapest-az only shows prices; it mustn't need DescribeSpotPriceHistory."""
    ec2 = _ec2(offered=USW2, subnets=USW2, price_error=client_error("UnauthorizedOperation"))
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


def _ctx(
    region: str, azs: tuple[str, ...], outcomes: dict[str | None, object], hard_pin: bool = False
) -> RegionContext:
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
        region=region,
        ec2_client=client,
        ami={"ImageId": "ami-1"},
        sg_id="sg-1",
        key_name="k",
        placement_azs=azs,
        hard_pin=hard_pin,
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


def test_hard_pin_unsupported_is_a_clean_region_named_error():
    """An EBS volume's / cluster's AZ is required: report AWS's reason as a CLIError (no traceback, no skip)."""
    contexts = {"us-west-2": _ctx("us-west-2", ("us-west-2d",), {"us-west-2d": UNSUPPORTED}, hard_pin=True)}
    with pytest.raises(CLIError, match=r"\[us-west-2\] g4dn.xlarge can't launch in us-west-2d: .*not supported"):
        launch_with_retry(LaunchConfig(regions=("us-west-2",)), lambda r: contexts[r])


@pytest.mark.parametrize("outcome,kind", [(UNSUPPORTED, "unsupported"), (PRICE, "price")])
def test_one_az_plan_failure_drops_the_region_and_tries_the_next(outcome, kind):
    """A plan that happens to hold one AZ (a single --az, or one usable AZ) is not a hard pin."""
    contexts = {
        "us-west-2": _ctx("us-west-2", ("us-west-2b",), {"us-west-2b": outcome}),
        "us-east-1": _ctx("us-east-1", (), {None: "i-east"}),
    }
    fatal: list[tuple[str, str]] = []
    result = launch_with_retry(
        LaunchConfig(regions=("us-west-2", "us-east-1")),
        lambda r: contexts[r],
        on_region_fatal=lambda region, k, msg: fatal.append((region, k)),
    )
    assert result.region == "us-east-1"
    assert fatal == [("us-west-2", kind)]


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
    assert "aws-bootstrap launch --dry-run --cheapest-az --instance-type g4dn.xlarge --region us-west-2" in msg


def test_failure_names_only_the_azs_actually_tried():
    contexts = {"us-west-2": _ctx("us-west-2", ("us-west-2b", "us-west-2a", "us-west-2c"), {"us-west-2b": QUOTA})}
    with pytest.raises(CLIError) as exc:
        # The on-demand fallback hits the same (region-wide) quota error in us-west-2b.
        launch_with_retry(LaunchConfig(regions=("us-west-2",)), lambda r: contexts[r], confirm_on_demand=lambda: True)
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


# ---------------------------------------------------------------------------
# Pool facts for the dry run: on-demand price, placement scores, vCPUs
# ---------------------------------------------------------------------------


def _product(*prices: str) -> str:
    """A Price List product (a JSON string, as get_products returns it) with these hourly prices."""
    dims = {f"dim{i}": {"unit": "Hrs", "pricePerUnit": {"USD": p}} for i, p in enumerate(prices)}
    return json.dumps({"terms": {"OnDemand": {"term": {"priceDimensions": dims}}}})


def _pricing_session(*, products=None, error=None, next_token=None) -> MagicMock:
    session = MagicMock()
    pricing = session.client.return_value
    if error:
        pricing.get_products.side_effect = error
    else:
        pricing.get_products.return_value = {
            "PriceList": list(products or []),
            **({"NextToken": next_token} if next_token else {}),
        }
    return session


def test_on_demand_price_asks_the_price_list_api_for_what_launch_runs():
    session = _pricing_session(products=[_product("3.3631200000")])
    assert on_demand_price(session, "g7e.2xlarge", "us-east-1") == 3.36312
    assert session.client.call_args.args == ("pricing",)
    assert session.client.call_args.kwargs["region_name"] == "us-east-1"
    config = session.client.call_args.kwargs["config"]  # bounded: it also runs after RunInstances
    assert (config.connect_timeout, config.read_timeout, config.retries["max_attempts"]) == (5, 10, 2)
    kwargs = session.client.return_value.get_products.call_args.kwargs
    assert kwargs["ServiceCode"] == "AmazonEC2"
    assert kwargs["MaxResults"] == 2  # what lets "exactly one product" be checked
    assert {f["Field"]: f["Value"] for f in kwargs["Filters"]} == {
        "instanceType": "g7e.2xlarge",
        "regionCode": "us-east-1",
        "operatingSystem": "Linux",
        "tenancy": "Shared",
        "preInstalledSw": "NA",
        "capacitystatus": "Used",
        "licenseModel": "No License required",
        "marketoption": "OnDemand",  # P types also have a Capacity Block product
    }


@pytest.mark.parametrize(
    "session",
    [
        pytest.param(_pricing_session(products=[]), id="no-product"),
        pytest.param(_pricing_session(products=[_product("1.0")], next_token="more"), id="more-on-a-next-page"),
        pytest.param(_pricing_session(products=['{"terms": {"OnDemand": []}}']), id="unexpected-shape"),
        pytest.param(_pricing_session(products=[_product("1.0"), _product("2.0")]), id="ambiguous-products"),
        pytest.param(_pricing_session(products=[_product("1.0", "2.0")]), id="ambiguous-prices"),
        pytest.param(_pricing_session(products=[_product("0.0000000000")]), id="zero-price"),
        pytest.param(_pricing_session(products=["not json"]), id="malformed"),
        pytest.param(_pricing_session(error=client_error("AccessDeniedException")), id="no-permission"),
        pytest.param(
            _pricing_session(error=botocore.exceptions.EndpointConnectionError(endpoint_url="https://api.pricing")),
            id="unreachable",
        ),
    ],
)
def test_on_demand_price_is_none_unless_exactly_one_price_matches(session):
    assert on_demand_price(session, "g7e.2xlarge", "us-east-1") is None


def _scores_client(records=None, error=None, zones=()) -> MagicMock:
    client = MagicMock()
    if error:
        client.get_paginator.return_value.paginate.side_effect = error
    else:
        client.get_paginator.return_value.paginate.return_value = [{"SpotPlacementScores": records or []}]
    client.describe_availability_zones.return_value = {
        "AvailabilityZones": [{"ZoneId": zone_id, "ZoneName": name} for zone_id, name in zones]
    }
    return client


def test_spot_placement_scores_map_zone_ids_to_this_accounts_zone_names():
    records = [
        {"Region": "us-east-1", "AvailabilityZoneId": "use1-az6", "Score": 9},
        {"Region": "us-east-1", "AvailabilityZoneId": "use1-az2", "Score": 1},
        {"Region": "us-east-1", "AvailabilityZoneId": "use1-az9", "Score": 5},  # unmapped: dropped
        {"Region": "us-west-2", "AvailabilityZoneId": "usw2-az1", "Score": 3},
    ]
    east = _scores_client(records, zones=[("use1-az6", "us-east-1c"), ("use1-az2", "us-east-1a")])
    west = _scores_client(zones=[("usw2-az1", "us-west-2b")])
    session = session_with_clients({"us-east-1": east, "us-west-2": west})
    assert spot_placement_scores(session, "g7e.2xlarge", ("us-east-1", "us-west-2")) == {
        "us-east-1": {"us-east-1c": 9, "us-east-1a": 1},
        "us-west-2": {"us-west-2b": 3},
    }
    east.get_paginator.assert_called_once_with("get_spot_placement_scores")
    first = session.client.call_args_list[0]  # the scores client is bounded like the pricing one
    assert first.kwargs["region_name"] == "us-east-1" and first.kwargs["config"].connect_timeout == 5
    assert east.get_paginator.return_value.paginate.call_args.kwargs == {
        "InstanceTypes": ["g7e.2xlarge"],
        "TargetCapacity": 1,
        "SingleAvailabilityZone": True,
        "RegionNames": ["us-east-1", "us-west-2"],
    }


@pytest.mark.parametrize(
    "regions,client",
    [
        pytest.param((), _scores_client(), id="no-regions"),
        pytest.param(("us-east-1",), _scores_client(error=client_error("UnauthorizedOperation")), id="no-permission"),
    ],
)
def test_spot_placement_scores_are_empty_when_unavailable(regions, client):
    assert spot_placement_scores(session_with_clients({"us-east-1": client}), "g7e.2xlarge", regions) == {}


def test_spot_placement_scores_skip_families_whose_single_type_scores_are_capped():
    """AWS caps single-type scores low outside P/G/Trn/Inf: a misleading score is worse than none."""
    client = _scores_client()
    assert spot_placement_scores(session_with_clients({"us-east-1": client}), "t3.large", ("us-east-1",)) == {}
    client.get_paginator.assert_not_called()


@pytest.mark.parametrize(
    "describe,expected",
    [
        pytest.param(
            {"return_value": {"InstanceTypes": [{"InstanceType": "g7e.2xlarge", "VCpuInfo": {"DefaultVCpus": 8}}]}},
            8,
            id="known",
        ),
        pytest.param({"return_value": {"InstanceTypes": []}}, None, id="unknown-type"),
        pytest.param({"side_effect": client_error("InvalidInstanceType")}, None, id="error"),
    ],
)
def test_instance_type_vcpus(describe, expected):
    ec2 = MagicMock()
    ec2.describe_instance_types.configure_mock(**describe)
    assert instance_type_vcpus(ec2, "g7e.2xlarge") == expected


def test_default_vcpus_asks_for_at_most_100_types_per_call():
    ec2 = MagicMock()
    ec2.describe_instance_types.side_effect = lambda InstanceTypes: {
        "InstanceTypes": [{"InstanceType": t, "VCpuInfo": {"DefaultVCpus": 4}} for t in InstanceTypes]
    }
    types = [f"g6.{i}xlarge" for i in range(150)]
    assert default_vcpus(ec2, [*types, types[0]]) == dict.fromkeys(types, 4)  # duplicates asked once
    assert [len(c.kwargs["InstanceTypes"]) for c in ec2.describe_instance_types.call_args_list] == [100, 50]
