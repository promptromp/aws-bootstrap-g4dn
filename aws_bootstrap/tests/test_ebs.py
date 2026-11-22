"""Tests for EBS data volume operations in ec2.py."""

from __future__ import annotations
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from aws_bootstrap.ec2 import (
    EBS_DEVICE_NAME,
    CLIError,
    VolumeLocation,
    attach_ebs_volume,
    create_ebs_volume,
    delete_ebs_volume,
    find_ebs_volumes_for_instance,
    find_orphan_ebs_volumes,
    find_orphan_ebs_volumes_in_regions,
    locate_ebs_volume,
    validate_ebs_volume,
)


@pytest.fixture
def ec2():
    """A bare mocked EC2 client."""
    return MagicMock()


def _client_error(code: str) -> botocore.exceptions.ClientError:
    return botocore.exceptions.ClientError({"Error": {"Code": code, "Message": "test"}}, "DescribeVolumes")


def _volume(state="available", az="us-west-2a", volume_id="vol-abc123", size=100) -> dict:
    return {"VolumeId": volume_id, "State": state, "AvailabilityZone": az, "Size": size}


def _tagged_volume(volume_id: str, size: int, instance_id: str | None) -> dict:
    """A detached tool-created volume; ``instance_id=None`` leaves out the instance link tag."""
    tags = [{"Key": "created-by", "Value": "aws-bootstrap-g4dn"}]
    if instance_id:
        tags.append({"Key": "aws-bootstrap-instance", "Value": instance_id})
    return {"VolumeId": volume_id, "Size": size, "State": "available", "Tags": tags}


# ---------------------------------------------------------------------------
# create / validate / attach / delete
# ---------------------------------------------------------------------------


def test_create_ebs_volume(ec2):
    ec2.create_volume.return_value = {"VolumeId": "vol-abc123"}

    assert create_ebs_volume(ec2, 96, "us-west-2a", "aws-bootstrap-g4dn", "i-test123") == "vol-abc123"

    ec2.create_volume.assert_called_once()
    create_kwargs = ec2.create_volume.call_args[1]
    assert create_kwargs["AvailabilityZone"] == "us-west-2a"
    assert create_kwargs["Size"] == 96
    assert create_kwargs["VolumeType"] == "gp3"
    tags = {t["Key"]: t["Value"] for t in create_kwargs["TagSpecifications"][0]["Tags"]}
    assert tags["created-by"] == "aws-bootstrap-g4dn"
    assert tags["Name"] == "aws-bootstrap-data-i-test123"
    assert tags["aws-bootstrap-instance"] == "i-test123"
    ec2.get_waiter.assert_called_once_with("volume_available")
    ec2.get_waiter.return_value.wait.assert_called_once()


def test_validate_ebs_volume_valid(ec2):
    ec2.describe_volumes.return_value = {"Volumes": [_volume()]}
    assert validate_ebs_volume(ec2, "vol-abc123", "us-west-2a")["VolumeId"] == "vol-abc123"


@pytest.mark.parametrize(
    "describe,match",
    [
        pytest.param({"return_value": {"Volumes": [_volume(az="us-east-1a")]}}, "us-east-1a", id="wrong-az"),
        pytest.param({"return_value": {"Volumes": [_volume(state="in-use")]}}, "in-use", id="in-use"),
        pytest.param({"side_effect": _client_error("InvalidVolume.NotFound")}, "not found", id="not-found"),
        pytest.param({"return_value": {"Volumes": []}}, "not found", id="empty-response"),
    ],
)
def test_validate_ebs_volume_rejects(ec2, describe, match):
    ec2.describe_volumes.configure_mock(**describe)
    with pytest.raises(CLIError, match=match):
        validate_ebs_volume(ec2, "vol-abc123", "us-west-2a")


@pytest.mark.parametrize(
    "kwargs,device",
    [({}, EBS_DEVICE_NAME), ({"device_name": "/dev/sdg"}, "/dev/sdg")],
    ids=["default-device", "custom-device"],
)
def test_attach_ebs_volume(ec2, kwargs, device):
    attach_ebs_volume(ec2, "vol-abc123", "i-test123", **kwargs)
    ec2.attach_volume.assert_called_once_with(VolumeId="vol-abc123", InstanceId="i-test123", Device=device)
    ec2.get_waiter.assert_called_once_with("volume_in_use")
    ec2.get_waiter.return_value.wait.assert_called_once()


def test_delete_ebs_volume(ec2):
    delete_ebs_volume(ec2, "vol-abc123")
    ec2.delete_volume.assert_called_once_with(VolumeId="vol-abc123")


# ---------------------------------------------------------------------------
# find_ebs_volumes_for_instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "volume,expected",
    [
        pytest.param(
            {
                "VolumeId": "vol-data1",
                "Size": 96,
                "State": "in-use",
                "Attachments": [{"Device": "/dev/sdf", "InstanceId": "i-test123"}],
            },
            {"VolumeId": "vol-data1", "Size": 96, "Device": "/dev/sdf", "State": "in-use"},
            id="attached",
        ),
        # Detached (available) volumes are still discovered by tags.
        pytest.param(
            {"VolumeId": "vol-avail", "Size": 50, "State": "available", "Attachments": []},
            {"VolumeId": "vol-avail", "Size": 50, "Device": "", "State": "available"},
            id="detached",
        ),
    ],
)
def test_find_ebs_volumes_for_instance(ec2, volume, expected):
    ec2.describe_volumes.return_value = {"Volumes": [volume]}
    volumes = find_ebs_volumes_for_instance(ec2, "i-test123", "aws-bootstrap-g4dn")
    assert len(volumes) == 1
    assert {k: volumes[0][k] for k in expected} == expected


@pytest.mark.parametrize(
    "describe",
    [
        pytest.param({"return_value": {"Volumes": []}}, id="no-volumes"),
        # ClientError (e.g. permissions) returns empty list instead of raising.
        pytest.param({"side_effect": _client_error("UnauthorizedOperation")}, id="client-error"),
    ],
)
def test_find_ebs_volumes_for_instance_empty(ec2, describe):
    ec2.describe_volumes.configure_mock(**describe)
    assert find_ebs_volumes_for_instance(ec2, "i-test123", "aws-bootstrap-g4dn") == []


# ---------------------------------------------------------------------------
# find_orphan_ebs_volumes
# ---------------------------------------------------------------------------


def test_find_orphan_ebs_volumes_returns_orphans(ec2):
    """Volumes whose linked instance is not live should be returned."""
    ec2.describe_volumes.return_value = {"Volumes": [_tagged_volume("vol-orphan1", 50, "i-dead1234")]}
    orphans = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids=set())
    assert len(orphans) == 1
    assert orphans[0]["VolumeId"] == "vol-orphan1"
    assert orphans[0]["InstanceId"] == "i-dead1234"
    assert orphans[0]["Size"] == 50
    # Only detached volumes are candidates: the API is asked for status=available.
    filter_names = {f["Name"] for f in ec2.describe_volumes.call_args[1]["Filters"]}
    assert "status" in filter_names


@pytest.mark.parametrize(
    "describe",
    [
        # Volumes linked to a live instance are not orphans.
        pytest.param({"return_value": {"Volumes": [_tagged_volume("vol-attached", 96, "i-live123")]}}, id="live"),
        pytest.param({"return_value": {"Volumes": []}}, id="no-volumes"),
        # Volumes without the aws-bootstrap-instance tag are skipped.
        pytest.param({"return_value": {"Volumes": [_tagged_volume("vol-notag", 10, None)]}}, id="no-instance-tag"),
        pytest.param({"side_effect": _client_error("UnauthorizedOperation")}, id="client-error"),
    ],
)
def test_find_orphan_ebs_volumes_none(ec2, describe):
    ec2.describe_volumes.configure_mock(**describe)
    assert find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids={"i-live123"}) == []


# ---------------------------------------------------------------------------
# find_orphan_ebs_volumes_in_regions
# ---------------------------------------------------------------------------


def _session(clients: dict[str, MagicMock]) -> MagicMock:
    session = MagicMock()
    session.client.side_effect = lambda service, region_name: clients[region_name]
    return session


def _orphan_client(*volumes: dict, error: Exception | None = None) -> MagicMock:
    client = MagicMock()
    if error:
        client.describe_volumes.side_effect = error
    else:
        client.describe_volumes.return_value = {"Volumes": list(volumes)}
    return client


def test_find_orphan_ebs_volumes_in_regions_labels_and_sorts():
    """Volumes from every region are merged, labelled with Region, and sorted."""
    session = _session(
        {
            "us-west-2": _orphan_client(_tagged_volume("vol-w", 50, "i-dead1")),
            "us-east-1": _orphan_client(_tagged_volume("vol-e", 100, "i-dead2")),
        }
    )
    orphans = find_orphan_ebs_volumes_in_regions(session, "aws-bootstrap-g4dn", ["us-west-2", "us-east-1"], set())
    assert [(v["VolumeId"], v["Region"]) for v in orphans] == [("vol-e", "us-east-1"), ("vol-w", "us-west-2")]


def test_find_orphan_ebs_volumes_in_regions_filters_live_instances():
    session = _session({"us-west-2": _orphan_client(_tagged_volume("vol-w", 50, "i-alive"))})
    assert find_orphan_ebs_volumes_in_regions(session, "aws-bootstrap-g4dn", ["us-west-2"], {"i-alive"}) == []


def test_find_orphan_ebs_volumes_in_regions_survives_region_failure():
    """A failing region contributes nothing but does not abort the sweep."""
    session = _session(
        {
            "eu-west-1": _orphan_client(error=_client_error("AuthFailure")),
            "us-west-2": _orphan_client(_tagged_volume("vol-w", 50, "i-dead1")),
        }
    )
    orphans = find_orphan_ebs_volumes_in_regions(session, "aws-bootstrap-g4dn", ["eu-west-1", "us-west-2"], set())
    assert [v["VolumeId"] for v in orphans] == ["vol-w"]


# ---------------------------------------------------------------------------
# locate_ebs_volume: an --ebs-volume-id implies its region and AZ
# ---------------------------------------------------------------------------

ATTACHED = {**_volume(state="in-use", az="us-east-1d", volume_id="vol-1"), "Attachments": [{"InstanceId": "i-old"}]}


def _volume_client(volume: dict | None = None, error: Exception | None = None) -> MagicMock:
    """describe_volumes in one region: *volume*, *error*, or (neither) InvalidVolume.NotFound."""
    client = MagicMock()
    if volume:
        client.describe_volumes.return_value = {"Volumes": [volume]}
    else:
        client.describe_volumes.side_effect = error or _client_error("InvalidVolume.NotFound")
    return client


def test_locate_finds_the_volume_in_the_first_regions_without_listing_others():
    everywhere = MagicMock()
    session = _session({"us-west-2": _volume_client(_volume(volume_id="vol-1"))})
    assert locate_ebs_volume(session, "vol-1", ("us-west-2",), everywhere) == VolumeLocation(
        "vol-1", "us-west-2", "us-west-2a", "available"
    )
    everywhere.assert_not_called()


def test_locate_then_searches_the_other_enabled_regions():
    clients = {"us-west-2": _volume_client(), "eu-west-1": _volume_client(), "us-east-1": _volume_client(ATTACHED)}
    loc = locate_ebs_volume(_session(clients), "vol-1", ("us-west-2",), lambda: ["eu-west-1", "us-east-1", "us-west-2"])
    assert loc == VolumeLocation("vol-1", "us-east-1", "us-east-1d", "in-use", attached_to="i-old")
    clients["us-west-2"].describe_volumes.assert_called_once()  # not searched twice


@pytest.mark.parametrize("code", ["InvalidParameterValue", "InvalidVolumeID.Malformed"])
def test_locate_rejects_a_malformed_id_without_searching_on(code):
    """AWS IDs carry a check value: a typo is malformed in every region, so stop at once."""
    everywhere = MagicMock()
    session = _session({"us-west-2": _volume_client(error=_client_error(code))})
    with pytest.raises(CLIError, match="vol-bogus is not a valid EBS volume ID; check it for a typo"):
        locate_ebs_volume(session, "vol-bogus", ("us-west-2",), everywhere)
    everywhere.assert_not_called()


@pytest.mark.parametrize("region_answering", ["us-west-2", "eu-west-1"], ids=["requested", "elsewhere"])
def test_locate_lets_bad_credentials_through(region_answering):
    """Bad credentials fail everywhere: they must reach the CLI's credentials message, not read as "not found"."""
    clients = {"us-west-2": _volume_client(), "eu-west-1": _volume_client()}
    clients[region_answering] = _volume_client(error=_client_error("AuthFailure"))
    with pytest.raises(botocore.exceptions.ClientError, match="AuthFailure"):
        locate_ebs_volume(_session(clients), "vol-1", ("us-west-2",), lambda: ["eu-west-1"])


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(_client_error("UnauthorizedOperation"), id="regional-deny"),
        pytest.param(botocore.exceptions.ConnectTimeoutError(endpoint_url="https://ec2"), id="unreachable"),
    ],
)
def test_locate_looks_past_a_requested_region_it_cannot_query(error):
    """An SCP denying the default region, or its endpoint timing out, mustn't hide a volume elsewhere."""
    clients = {"us-west-2": _volume_client(error=error), "us-east-1": _volume_client(ATTACHED)}
    loc = locate_ebs_volume(_session(clients), "vol-1", ("us-west-2",), lambda: ["us-east-1", "us-west-2"])
    assert loc.region == "us-east-1"


def _cannot_list() -> list[str]:
    raise _client_error("UnauthorizedOperation")


@pytest.mark.parametrize(
    "clients,everywhere,match",
    [
        pytest.param(
            {"us-west-2": _volume_client(), "us-east-1": _volume_client()},
            lambda: ["us-east-1", "us-west-2"],
            r"vol-1 not found in the 2 region\(s\) checked\.$",
            id="nowhere",
        ),
        pytest.param(
            {"us-west-2": _volume_client(), "eu-west-1": _volume_client(error=_client_error("UnauthorizedOperation"))},
            lambda: ["eu-west-1"],
            r"not found in the 1 region\(s\) checked\.\n  Couldn't check: eu-west-1 \(UnauthorizedOperation\)$",
            id="unqueryable-region-is-named",
        ),
        pytest.param(
            {
                "us-west-2": _volume_client(),
                "eu-west-1": _volume_client(error=botocore.exceptions.ReadTimeoutError(endpoint_url="https://x")),
            },
            lambda: ["eu-west-1"],
            r"Couldn't check: eu-west-1 \(ReadTimeoutError\)$",
            id="timed-out-region-is-named",
        ),
        pytest.param(
            {
                "us-west-2": _volume_client(error=_client_error("UnauthorizedOperation")),
                "eu-west-1": _volume_client(error=_client_error("UnauthorizedOperation")),
                "us-east-1": _volume_client(error=_client_error("OptInRequired")),
            },
            lambda: ["eu-west-1", "us-east-1"],
            r"no region answered\.\n  Couldn't check: us-east-1 \(OptInRequired\); "
            r"eu-west-1, us-west-2 \(UnauthorizedOperation\)$",
            id="nothing-answered-grouped-by-reason",
        ),
        pytest.param(
            {"us-west-2": _volume_client()},
            _cannot_list,
            r"Couldn't check: other regions \(could not list them: UnauthorizedOperation\)$",
            id="region-list-unavailable",
        ),
    ],
)
def test_locate_not_found_says_what_was_and_wasnt_searched(clients, everywhere, match):
    with pytest.raises(CLIError, match=match):
        locate_ebs_volume(_session(clients), "vol-1", ("us-west-2",), everywhere)
