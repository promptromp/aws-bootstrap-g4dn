"""CliRunner tests for the `cluster` command group (Phase 1)."""

from __future__ import annotations
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.ec2 import RegionContext, RegionLaunch


def _runner():
    return CliRunner()


def _node(instance_id="i-1", cluster_id="ml1", rank=0, state="running"):
    return {
        "InstanceId": instance_id,
        "ClusterId": cluster_id,
        "Rank": rank,
        "State": state,
        "InstanceType": "g5.xlarge",
        "PublicIp": "1.2.3.4",
        "PrivateIp": "10.0.0.5",
        "AvailabilityZone": "us-east-1c",
        "Lifecycle": "spot",
        "LaunchTime": datetime(2026, 5, 21),
    }


def test_cluster_help_lists_subcommands():
    result = _runner().invoke(main, ["cluster", "--help"])
    assert result.exit_code == 0
    for sub in ("launch", "status", "terminate"):
        assert sub in result.output


def test_cluster_status_json_lists_nodes():
    with (
        patch("aws_bootstrap.cli.boto3.Session"),
        patch("aws_bootstrap.cli.find_cluster_instances", return_value=[_node()]),
    ):
        result = _runner().invoke(main, ["-o", "json", "cluster", "status", "--cluster-id", "ml1"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["nodes"][0]["rank"] == 0
    assert data["nodes"][0]["instance_id"] == "i-1"


def test_cluster_terminate_requires_yes_in_json_mode():
    with (
        patch("aws_bootstrap.cli.boto3.Session"),
        patch("aws_bootstrap.cli.find_cluster_instances", return_value=[_node()]),
    ):
        result = _runner().invoke(main, ["-o", "json", "cluster", "terminate", "--cluster-id", "ml1"])
    assert result.exit_code != 0


@patch("aws_bootstrap.cli.delete_cluster_placement_group")
@patch("aws_bootstrap.cli.remove_ssh_host", return_value="aws-ml1-0")
@patch("aws_bootstrap.cli.terminate_tagged_instances", return_value=[])
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_terminate_tears_down_nodes_and_group(mock_session, mock_find, mock_term, mock_remove, mock_delpg):
    mock_find.return_value = [_node()]
    result = _runner().invoke(main, ["cluster", "terminate", "--cluster-id", "ml1", "--yes", "--keep-ebs"])
    assert result.exit_code == 0, result.output
    mock_term.assert_called_once()
    assert mock_term.call_args[0][1] == ["i-1"]
    mock_delpg.assert_called_once()


@patch("aws_bootstrap.cli.add_ssh_host", return_value="aws-ml1-0")
@patch("aws_bootstrap.cli.wait_instance_ready")
@patch("aws_bootstrap.cli.launch_with_retry")
@patch("aws_bootstrap.cli.ensure_cluster_security_group_rule")
@patch("aws_bootstrap.cli.ensure_cluster_placement_group", return_value="aws-bootstrap-cluster-ml1")
@patch("aws_bootstrap.cli.ensure_security_group", return_value="sg-123")
@patch("aws_bootstrap.cli.import_key_pair", return_value="aws-bootstrap-key")
@patch("aws_bootstrap.cli.get_latest_ami", return_value={"ImageId": "ami-1", "Name": "DL"})
@patch("aws_bootstrap.cli.find_cluster_instances", return_value=[])
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_launch_two_nodes_tags_ranks(
    mock_session,
    mock_find,
    mock_ami,
    mock_import,
    mock_sg,
    mock_pg,
    mock_sgrule,
    mock_launch,
    mock_wait,
    mock_add_ssh,
    tmp_path,
):
    key = tmp_path / "id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAA test@host")

    def _launch(config, prepare_region, **kw):
        ctx = RegionContext(
            region="us-east-1",
            ec2_client=MagicMock(),
            ami={"ImageId": "ami-1"},
            sg_id="sg-123",
            key_name="aws-bootstrap-key",
            placement_az="us-east-1c",
            placement_group="aws-bootstrap-cluster-ml1",
        )
        return RegionLaunch("us-east-1", ctx, {"InstanceId": "i-x"}, "spot")

    mock_launch.side_effect = _launch
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-east-1c"}}

    result = _runner().invoke(
        main,
        [
            "cluster",
            "launch",
            "--cluster-id",
            "ml1",
            "--nodes",
            "2",
            "--region",
            "us-east-1",
            "--key-path",
            str(key),
            "--no-setup",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mock_launch.call_count == 2
    aliases = {c.kwargs.get("alias") for c in mock_add_ssh.call_args_list}
    assert aliases == {"aws-ml1-0", "aws-ml1-1"}
