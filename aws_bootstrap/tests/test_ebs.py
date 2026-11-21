"""Tests for EBS data volume operations in ec2.py."""

from __future__ import annotations
from unittest.mock import MagicMock

import botocore.exceptions
import pytest

from aws_bootstrap.ec2 import (
    EBS_DEVICE_NAME,
    CLIError,
    attach_ebs_volume,
    create_ebs_volume,
    delete_ebs_volume,
    find_ebs_volumes_for_instance,
    find_orphan_ebs_volumes,
    find_orphan_ebs_volumes_in_regions,
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


@pytest.mark.parametrize("kwargs,device", [({}, EBS_DEVICE_NAME), ({"device_name": "/dev/sdg"}, "/dev/sdg")])
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
