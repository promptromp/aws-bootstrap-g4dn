"""Pre-launch checks wired into launch: --iam-instance-profile and --ebs-volume-id.

Both run read-only before anything is created; the volume also decides the region.
(cluster launch's profile wiring is in test_cli_cluster.py.)
"""

from __future__ import annotations
import json
from unittest.mock import patch

import pytest
import yaml

from aws_bootstrap.cli import main
from aws_bootstrap.ec2 import CLIError, VolumeLocation


EAST_VOLUME = VolumeLocation("vol-1", "us-east-1", "us-east-1d", "available")


def _launch(runner, key_path, *args: str, fmt: str = "text"):
    return runner.invoke(main, ["-o", fmt, "launch", "--key-path", str(key_path), *args])


# ---------------------------------------------------------------------------
# --iam-instance-profile
# ---------------------------------------------------------------------------


def test_profile_check_fails_before_anything_regional_happens(runner, launch_aws, key_path):
    launch_aws.profile_check.side_effect = CLIError("IAM instance profile 'nope' not found.")
    result = _launch(runner, key_path, "--dry-run", "--iam-instance-profile", "nope")
    assert result.exit_code != 0
    assert "'nope' not found" in result.output
    launch_aws.profile_check.assert_called_once()
    assert launch_aws.profile_check.call_args[0][1] == "nope"
    # Only the global IAM client was built: no AMI lookup, key import or security group.
    assert {c.args[0] for c in launch_aws.session.return_value.client.call_args_list} == {"iam"}
    launch_aws.ami.assert_not_called()
    launch_aws.import_key.assert_not_called()
    launch_aws.sg.assert_not_called()


def test_no_profile_means_no_iam_call(runner, launch_aws, key_path):
    assert _launch(runner, key_path, "--dry-run").exit_code == 0
    launch_aws.profile_check.assert_not_called()


@pytest.mark.parametrize(
    "roles,label",
    [(["app-role"], "app (role app-role)"), (None, "app (not verified)"), ([], "app (no role)")],
    ids=["verified", "unverifiable", "roleless"],
)
def test_dry_run_text_shows_the_profile(runner, launch_aws, key_path, roles, label):
    launch_aws.profile_check.return_value = roles
    result = _launch(runner, key_path, "--dry-run", "--iam-instance-profile", "app")
    assert result.exit_code == 0, result.output
    assert "IAM instance profile" in result.output
    assert label in result.output


@pytest.mark.parametrize("fmt,load", [("json", json.loads), ("yaml", yaml.safe_load), ("table", None)])
def test_dry_run_structured_output_names_the_profile(runner, launch_aws, key_path, fmt, load):
    arn = "arn:aws:iam::123456789012:instance-profile/team/app"
    result = _launch(runner, key_path, "--dry-run", "--iam-instance-profile", arn, fmt=fmt)
    assert result.exit_code == 0, result.output
    if load:
        assert load(result.output)["iam_instance_profile"] == "app"
    else:
        assert "iam_instance_profile" in result.output and "app" in result.output


def test_launch_threads_the_profile_into_the_launch_and_its_result(runner, launched, key_path):
    result = _launch(runner, key_path, "--no-setup", "--iam-instance-profile", "app", fmt="json")
    assert result.exit_code == 0, result.output
    assert launched.launch.call_args[0][0].iam_instance_profile == "app"
    assert json.loads(result.output)["iam_instance_profile"] == "app"


def test_incomplete_launch_record_names_the_profile(runner, launched, key_path):
    launched.ready.return_value = {"Placement": {"AvailabilityZone": "us-west-2a"}}  # no public IP
    result = _launch(runner, key_path, "--no-setup", "--iam-instance-profile", "app", fmt="json")
    assert result.exit_code != 0
    record = json.loads(result.output)
    assert record["status"] == "no-public-ip"
    assert record["iam_instance_profile"] == "app"


# ---------------------------------------------------------------------------
# --ebs-volume-id: the volume implies its region and AZ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regions,searched_first",
    [([], ("us-west-2",)), (["--region", "us-east-1", "--region", "us-west-2"], ("us-east-1", "us-west-2"))],
    ids=["default-region", "explicit-regions"],
)
def test_volume_moves_the_launch_to_its_region(runner, launch_aws, key_path, regions, searched_first):
    launch_aws.volume.side_effect = None
    launch_aws.volume.return_value = EAST_VOLUME
    result = _launch(runner, key_path, "--dry-run", "--ebs-volume-id", "vol-1", *regions, fmt="json")
    assert result.exit_code == 0, result.output
    assert launch_aws.volume.call_args[0][1:3] == ("vol-1", searched_first)
    out = json.loads(result.output)
    assert out["regions"] == ["us-east-1"]
    assert out["ebs_volume_id"] == "vol-1"
    assert out["ebs_volume_availability_zone"] == "us-east-1d"


def test_volume_region_inference_is_announced_in_text(runner, launch_aws, key_path):
    launch_aws.volume.side_effect = None
    launch_aws.volume.return_value = EAST_VOLUME
    result = _launch(runner, key_path, "--dry-run", "--ebs-volume-id", "vol-1")
    assert result.exit_code == 0, result.output
    assert "vol-1 is in us-east-1d → launching in us-east-1" in result.output
    assert "vol-1 (existing, us-east-1d, mounted at /data)" in result.output


def test_volume_search_falls_back_to_every_enabled_region(runner, launch_aws, key_path):
    _launch(runner, key_path, "--dry-run", "--ebs-volume-id", "vol-1")
    everywhere = launch_aws.volume.call_args[0][3]
    with patch("aws_bootstrap.cli.list_enabled_regions", return_value=["eu-west-1", "us-east-1"]) as enabled:
        assert everywhere() == ["eu-west-1", "us-east-1"]
    enabled.assert_called_once()


@pytest.mark.parametrize(
    "volume,regions,match",
    [
        pytest.param(
            EAST_VOLUME,
            ["--region", "us-west-2"],
            "vol-1 is in us-east-1 (us-east-1d), not in --region us-west-2",
            id="outside-explicit-region",
        ),
        pytest.param(
            VolumeLocation("vol-1", "us-west-2", "us-west-2a", "in-use", attached_to="i-old"),
            [],
            "is 'in-use' (attached to i-old); it must be 'available'",
            id="still-attached",
        ),
    ],
)
def test_volume_problems_fail_before_anything_is_created(runner, launch_aws, key_path, volume, regions, match):
    launch_aws.volume.side_effect = None
    launch_aws.volume.return_value = volume
    with patch("aws_bootstrap.cli.launch_with_retry") as launch:
        result = _launch(runner, key_path, "--ebs-volume-id", "vol-1", *regions)
    assert result.exit_code != 0
    assert match in result.output
    launch.assert_not_called()
    launch_aws.import_key.assert_not_called()
    launch_aws.sg.assert_not_called()
