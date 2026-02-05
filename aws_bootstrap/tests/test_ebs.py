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
    detach_ebs_volume,
    find_ebs_volumes_for_instance,
    find_orphan_ebs_volumes,
    validate_ebs_volume,
)


# ---------------------------------------------------------------------------
# create_ebs_volume
# ---------------------------------------------------------------------------


def test_create_ebs_volume():
    ec2 = MagicMock()
    ec2.create_volume.return_value = {"VolumeId": "vol-abc123"}
    waiter = MagicMock()
    ec2.get_waiter.return_value = waiter

    vol_id = create_ebs_volume(ec2, 96, "us-west-2a", "aws-bootstrap-g4dn", "i-test123")

    assert vol_id == "vol-abc123"
    ec2.create_volume.assert_called_once()
    create_kwargs = ec2.create_volume.call_args[1]
    assert create_kwargs["AvailabilityZone"] == "us-west-2a"
    assert create_kwargs["Size"] == 96
    assert create_kwargs["VolumeType"] == "gp3"

    # Check tags
    tags = create_kwargs["TagSpecifications"][0]["Tags"]
    tag_dict = {t["Key"]: t["Value"] for t in tags}
    assert tag_dict["created-by"] == "aws-bootstrap-g4dn"
    assert tag_dict["Name"] == "aws-bootstrap-data-i-test123"
    assert tag_dict["aws-bootstrap-instance"] == "i-test123"

    ec2.get_waiter.assert_called_once_with("volume_available")
    waiter.wait.assert_called_once()


# ---------------------------------------------------------------------------
# validate_ebs_volume
# ---------------------------------------------------------------------------


def test_validate_ebs_volume_valid():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-abc123",
                "State": "available",
                "AvailabilityZone": "us-west-2a",
                "Size": 100,
            }
        ]
    }
    vol = validate_ebs_volume(ec2, "vol-abc123", "us-west-2a")
    assert vol["VolumeId"] == "vol-abc123"


def test_validate_ebs_volume_wrong_az():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-abc123",
                "State": "available",
                "AvailabilityZone": "us-east-1a",
                "Size": 100,
            }
        ]
    }
    with pytest.raises(CLIError, match="us-east-1a"):
        validate_ebs_volume(ec2, "vol-abc123", "us-west-2a")


def test_validate_ebs_volume_in_use():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-abc123",
                "State": "in-use",
                "AvailabilityZone": "us-west-2a",
                "Size": 100,
            }
        ]
    }
    with pytest.raises(CLIError, match="in-use"):
        validate_ebs_volume(ec2, "vol-abc123", "us-west-2a")


def test_validate_ebs_volume_not_found():
    ec2 = MagicMock()
    ec2.describe_volumes.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "InvalidVolume.NotFound", "Message": "not found"}},
        "DescribeVolumes",
    )
    with pytest.raises(CLIError, match="not found"):
        validate_ebs_volume(ec2, "vol-notfound", "us-west-2a")


def test_validate_ebs_volume_empty_response():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {"Volumes": []}
    with pytest.raises(CLIError, match="not found"):
        validate_ebs_volume(ec2, "vol-empty", "us-west-2a")


# ---------------------------------------------------------------------------
# attach_ebs_volume
# ---------------------------------------------------------------------------


def test_attach_ebs_volume():
    ec2 = MagicMock()
    waiter = MagicMock()
    ec2.get_waiter.return_value = waiter

    attach_ebs_volume(ec2, "vol-abc123", "i-test123")

    ec2.attach_volume.assert_called_once_with(
        VolumeId="vol-abc123",
        InstanceId="i-test123",
        Device=EBS_DEVICE_NAME,
    )
    ec2.get_waiter.assert_called_once_with("volume_in_use")
    waiter.wait.assert_called_once()


def test_attach_ebs_volume_custom_device():
    ec2 = MagicMock()
    waiter = MagicMock()
    ec2.get_waiter.return_value = waiter

    attach_ebs_volume(ec2, "vol-abc123", "i-test123", device_name="/dev/sdg")

    ec2.attach_volume.assert_called_once_with(
        VolumeId="vol-abc123",
        InstanceId="i-test123",
        Device="/dev/sdg",
    )


# ---------------------------------------------------------------------------
# detach_ebs_volume
# ---------------------------------------------------------------------------


def test_detach_ebs_volume():
    ec2 = MagicMock()
    waiter = MagicMock()
    ec2.get_waiter.return_value = waiter

    detach_ebs_volume(ec2, "vol-abc123")

    ec2.detach_volume.assert_called_once_with(VolumeId="vol-abc123")
    ec2.get_waiter.assert_called_once_with("volume_available")
    waiter.wait.assert_called_once()


# ---------------------------------------------------------------------------
# delete_ebs_volume
# ---------------------------------------------------------------------------


def test_delete_ebs_volume():
    ec2 = MagicMock()
    delete_ebs_volume(ec2, "vol-abc123")
    ec2.delete_volume.assert_called_once_with(VolumeId="vol-abc123")


# ---------------------------------------------------------------------------
# find_ebs_volumes_for_instance
# ---------------------------------------------------------------------------


def test_find_ebs_volumes_for_instance():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-data1",
                "Size": 96,
                "State": "in-use",
                "Attachments": [{"Device": "/dev/sdf", "InstanceId": "i-test123"}],
            }
        ]
    }
    volumes = find_ebs_volumes_for_instance(ec2, "i-test123", "aws-bootstrap-g4dn")
    assert len(volumes) == 1
    assert volumes[0]["VolumeId"] == "vol-data1"
    assert volumes[0]["Size"] == 96
    assert volumes[0]["Device"] == "/dev/sdf"
    assert volumes[0]["State"] == "in-use"


def test_find_ebs_volumes_empty():
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {"Volumes": []}
    volumes = find_ebs_volumes_for_instance(ec2, "i-test123", "aws-bootstrap-g4dn")
    assert volumes == []


def test_find_ebs_volumes_includes_available():
    """Detached (available) volumes are still discovered by tags."""
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-avail",
                "Size": 50,
                "State": "available",
                "Attachments": [],
            }
        ]
    }
    volumes = find_ebs_volumes_for_instance(ec2, "i-old", "aws-bootstrap-g4dn")
    assert len(volumes) == 1
    assert volumes[0]["VolumeId"] == "vol-avail"
    assert volumes[0]["State"] == "available"
    assert volumes[0]["Device"] == ""


def test_find_ebs_volumes_client_error_returns_empty():
    """ClientError (e.g. permissions) returns empty list instead of raising."""
    ec2 = MagicMock()
    ec2.describe_volumes.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "no access"}},
        "DescribeVolumes",
    )
    volumes = find_ebs_volumes_for_instance(ec2, "i-test", "aws-bootstrap-g4dn")
    assert volumes == []


# ---------------------------------------------------------------------------
# find_orphan_ebs_volumes
# ---------------------------------------------------------------------------


def test_find_orphan_ebs_volumes_returns_orphans():
    """Volumes whose linked instance is not live should be returned."""
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-orphan1",
                "Size": 50,
                "State": "available",
                "Tags": [
                    {"Key": "created-by", "Value": "aws-bootstrap-g4dn"},
                    {"Key": "aws-bootstrap-instance", "Value": "i-dead1234"},
                ],
            }
        ]
    }
    orphans = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids=set())
    assert len(orphans) == 1
    assert orphans[0]["VolumeId"] == "vol-orphan1"
    assert orphans[0]["InstanceId"] == "i-dead1234"
    assert orphans[0]["Size"] == 50

    # Verify the API was called with status=available filter
    filters = ec2.describe_volumes.call_args[1]["Filters"]
    filter_names = {f["Name"] for f in filters}
    assert "status" in filter_names


def test_find_orphan_ebs_volumes_excludes_live_instances():
    """Volumes linked to a live instance should NOT be returned."""
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-attached",
                "Size": 96,
                "State": "available",
                "Tags": [
                    {"Key": "created-by", "Value": "aws-bootstrap-g4dn"},
                    {"Key": "aws-bootstrap-instance", "Value": "i-live123"},
                ],
            }
        ]
    }
    orphans = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids={"i-live123"})
    assert orphans == []


def test_find_orphan_ebs_volumes_empty():
    """No volumes at all should return empty list."""
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {"Volumes": []}
    orphans = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids=set())
    assert orphans == []


def test_find_orphan_ebs_volumes_skips_no_instance_tag():
    """Volumes without aws-bootstrap-instance tag should be skipped."""
    ec2 = MagicMock()
    ec2.describe_volumes.return_value = {
        "Volumes": [
            {
                "VolumeId": "vol-notag",
                "Size": 10,
                "State": "available",
                "Tags": [{"Key": "created-by", "Value": "aws-bootstrap-g4dn"}],
            }
        ]
    }
    orphans = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids=set())
    assert orphans == []


def test_find_orphan_ebs_volumes_client_error():
    """ClientError should return empty list."""
    ec2 = MagicMock()
    ec2.describe_volumes.side_effect = botocore.exceptions.ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "no access"}},
        "DescribeVolumes",
    )
    orphans = find_orphan_ebs_volumes(ec2, "aws-bootstrap-g4dn", live_instance_ids=set())
    assert orphans == []
