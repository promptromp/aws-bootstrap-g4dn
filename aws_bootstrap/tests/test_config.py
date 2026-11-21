"""Tests for LaunchConfig defaults and overrides."""

from __future__ import annotations
from pathlib import Path

import pytest

from aws_bootstrap.config import DEFAULT_REGION, LaunchConfig


def test_defaults():
    config = LaunchConfig()
    assert config.instance_type == "g4dn.xlarge"
    assert config.region == "us-west-2"
    assert config.spot is True
    assert config.volume_size == 100
    assert config.ssh_user == "ubuntu"
    assert config.key_name == "aws-bootstrap-key"
    assert config.security_group == "aws-bootstrap-ssh"
    assert config.tag_value == "aws-bootstrap-g4dn"
    assert config.run_setup is True
    assert config.dry_run is False
    assert config.ebs_storage is None
    assert config.ebs_volume_id is None
    assert config.wait is False
    assert config.wait_timeout == 1800


def test_overrides():
    config = LaunchConfig(
        instance_type="g5.xlarge",
        regions=("us-east-1",),
        spot=False,
        volume_size=200,
        key_path=Path("/tmp/test.pub"),
    )
    assert config.instance_type == "g5.xlarge"
    assert config.region == "us-east-1"
    assert config.spot is False
    assert config.volume_size == 200
    assert config.key_path == Path("/tmp/test.pub")


@pytest.mark.parametrize(
    "kwargs,storage,volume_id",
    [({"ebs_storage": 96}, 96, None), ({"ebs_volume_id": "vol-abc123"}, None, "vol-abc123")],
    ids=["ebs-storage", "ebs-volume-id"],
)
def test_ebs_overrides_leave_the_other_unset(kwargs, storage, volume_id):
    config = LaunchConfig(**kwargs)
    assert config.ebs_storage == storage
    assert config.ebs_volume_id == volume_id


def test_region_property_returns_first_of_regions():
    config = LaunchConfig(regions=("us-east-1", "us-west-2", "eu-west-1"))
    assert config.region == "us-east-1"
    assert config.regions == ("us-east-1", "us-west-2", "eu-west-1")


def test_region_property_falls_back_when_regions_empty():
    # Latent-invariant guard: never raise an opaque IndexError.
    assert LaunchConfig(regions=()).region == DEFAULT_REGION
