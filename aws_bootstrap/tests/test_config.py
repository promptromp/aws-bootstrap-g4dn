"""Tests for LaunchConfig defaults and overrides."""

from __future__ import annotations
from pathlib import Path

from aws_bootstrap.config import LaunchConfig


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


def test_overrides():
    config = LaunchConfig(
        instance_type="g5.xlarge",
        region="us-east-1",
        spot=False,
        volume_size=200,
        key_path=Path("/tmp/test.pub"),
    )
    assert config.instance_type == "g5.xlarge"
    assert config.region == "us-east-1"
    assert config.spot is False
    assert config.volume_size == 200
    assert config.key_path == Path("/tmp/test.pub")
