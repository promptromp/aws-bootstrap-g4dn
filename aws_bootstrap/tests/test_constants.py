"""Constants module: single-source invariants and sane values."""

from __future__ import annotations

from aws_bootstrap import config, constants, ec2
from aws_bootstrap.config import LaunchConfig


def test_constants_reexport_config_defaults():
    assert constants.DEFAULT_REGION == config.DEFAULT_REGION
    assert constants.DEFAULT_WAIT_TIMEOUT == config.DEFAULT_WAIT_TIMEOUT
    assert constants.TAG_VALUE == config.DEFAULT_TAG_VALUE
    assert constants.ALIAS_PREFIX == config.DEFAULT_ALIAS_PREFIX
    assert constants.SSH_PORT_DEFAULT == config.DEFAULT_SSH_PORT


def test_launchconfig_defaults_track_constants():
    c = LaunchConfig()
    assert c.tag_value == constants.TAG_VALUE
    assert c.alias_prefix == constants.ALIAS_PREFIX
    assert c.ssh_port == constants.SSH_PORT_DEFAULT


def test_ec2_still_exposes_ebs_device_name():
    # EBS_DEVICE_NAME stays importable from ec2 (used as a default there);
    # EBS_MOUNT_POINT's canonical home is now constants.
    assert ec2.EBS_DEVICE_NAME == constants.EBS_DEVICE_NAME


def test_waiter_profiles_distinct_and_documented():
    # Attach/availability is the fast profile; detach-on-terminate is slower.
    assert constants.EBS_VOLUME_WAITER == {"Delay": 5, "MaxAttempts": 24}
    assert constants.EBS_DETACH_WAITER == {"Delay": 10, "MaxAttempts": 30}
    assert constants.EBS_DETACH_WAITER != constants.EBS_VOLUME_WAITER


def test_ami_owner_ids_are_vendor_accounts():
    # Canonical's public Ubuntu AMI account — not a personal account.
    assert constants.AMI_OWNER_CANONICAL == "099720109477"
    assert constants.AMI_OWNER_AMAZON == "amazon"
