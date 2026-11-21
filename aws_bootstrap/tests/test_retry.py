"""Tests for region resolution and bounded-backoff helpers."""

from __future__ import annotations
import random

import pytest

from aws_bootstrap.retry import (
    BACKOFF_CAP,
    BACKOFF_JITTER,
    backoff_sleep_seconds,
    group_azs_by_region,
    parse_duration,
    region_of_az,
    resolve_regions,
)


# --- resolve_regions ---------------------------------------------------------


def test_resolve_regions_explicit_wins_and_preserves_order():
    assert resolve_regions(("us-east-1", "us-west-2"), "eu-west-1") == ("us-east-1", "us-west-2")


def test_resolve_regions_falls_back_to_session_region():
    assert resolve_regions((), "eu-west-1") == ("eu-west-1",)


def test_resolve_regions_falls_back_to_default():
    assert resolve_regions((), None) == ("us-west-2",)


def test_resolve_regions_explicit_single():
    assert resolve_regions(("ap-south-1",), None) == ("ap-south-1",)


# --- parse_duration ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("90s", 90),
        ("30m", 1800),
        ("1h", 3600),
        ("3600", 3600),
        ("  45 m ", 2700),
        ("2H", 7200),
    ],
)
def test_parse_duration_valid(value, expected):
    assert parse_duration(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "-5", "0", "10d", "1.5h", "0s"])
def test_parse_duration_invalid(value):
    with pytest.raises(ValueError):
        parse_duration(value)


# --- backoff_sleep_seconds ---------------------------------------------------


def test_backoff_grows_exponentially_without_jitter():
    no_jitter = [backoff_sleep_seconds(i, jitter=0.0) for i in range(4)]
    assert no_jitter == [30.0, 60.0, 120.0, 240.0]


def test_backoff_is_capped():
    # Large attempt index without jitter must not exceed the cap.
    assert backoff_sleep_seconds(20, jitter=0.0) == BACKOFF_CAP


def test_backoff_jitter_within_bounds_and_deterministic_with_seed():
    rng = random.Random(1234)
    values = [backoff_sleep_seconds(2, rng=rng) for _ in range(50)]
    raw = min(30.0 * (2.0**2), BACKOFF_CAP)
    for v in values:
        assert raw * (1 - BACKOFF_JITTER) <= v <= raw * (1 + BACKOFF_JITTER)
    # Same seed reproduces the same sequence.
    rng2 = random.Random(1234)
    assert [backoff_sleep_seconds(2, rng=rng2) for _ in range(50)] == values


def test_backoff_never_negative():
    assert backoff_sleep_seconds(0, jitter=1.0, rng=random.Random(0)) >= 0.0


# --- region_of_az / group_azs_by_region --------------------------------------


@pytest.mark.parametrize(
    "az,region",
    [
        ("us-west-2b", "us-west-2"),
        ("ap-southeast-1c", "ap-southeast-1"),
        ("il-central-1a", "il-central-1"),
        ("us-gov-west-1a", "us-gov-west-1"),
        ("us-west-2-lax-1a", "us-west-2"),  # Local Zone
        ("us-east-1-wl1-bos-wlz-1", "us-east-1"),  # Wavelength Zone
    ],
)
def test_region_of_az(az, region):
    assert region_of_az(az) == region


@pytest.mark.parametrize("bad", ["us-west-2", "useast1a", "", "us-west-2B"])
def test_region_of_az_rejects_non_az_names(bad):
    with pytest.raises(ValueError, match="Not an availability zone"):
        region_of_az(bad)


def test_group_azs_without_regions_uses_the_azs_regions_in_first_seen_order():
    regions, by_region = group_azs_by_region((), ("us-west-2b", "us-east-1c", "us-west-2a", "us-west-2b"))
    assert regions == ("us-west-2", "us-east-1")
    assert by_region == {"us-west-2": ("us-west-2b", "us-west-2a"), "us-east-1": ("us-east-1c",)}


def test_group_azs_keeps_explicit_region_order():
    regions, by_region = group_azs_by_region(("us-east-1", "us-west-2"), ("us-west-2b",))
    assert regions == ("us-east-1", "us-west-2")
    assert by_region == {"us-west-2": ("us-west-2b",)}


def test_group_azs_rejects_an_az_outside_the_given_regions():
    with pytest.raises(ValueError, match="us-east-1c is not in any --region"):
        group_azs_by_region(("us-west-2",), ("us-east-1c",))


def test_group_azs_no_azs_is_a_no_op():
    assert group_azs_by_region(("us-west-2",), ()) == (("us-west-2",), {})
    assert group_azs_by_region((), ()) == ((), {})
