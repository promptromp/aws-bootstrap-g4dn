"""CliRunner tests for the `cluster` command group (Phases 1-2)."""

from __future__ import annotations
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from aws_bootstrap.cli import main
from aws_bootstrap.cluster import NodeResult
from aws_bootstrap.ec2 import RegionContext, RegionLaunch
from aws_bootstrap.gpu import GpuInfo


def _runner():
    return CliRunner()


def _make_key(tmp_path):
    key = tmp_path / "id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAA test@host")
    return key


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


# ---------------------------------------------------------------------------
# cluster test / prepare (Phase 2)
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.cluster_mod.run_canary")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_test_runs_canary(mock_session, mock_find, mock_canary, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_canary.return_value = [
        NodeResult("i-0", 0, 0, "[canary] ok", ""),
        NodeResult("i-1", 1, 0, "[canary] ok", ""),
    ]
    result = _runner().invoke(
        main, ["-o", "json", "cluster", "test", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["passed"] is True
    mock_canary.assert_called_once()


@patch("aws_bootstrap.cli.cluster_mod.run_canary")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_test_nonzero_on_failure(mock_session, mock_find, mock_canary, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0)]
    mock_canary.return_value = [NodeResult("i-0", 0, 1, "", "boom")]
    result = _runner().invoke(
        main, ["-o", "json", "cluster", "test", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))]
    )
    assert result.exit_code != 0
    assert json.loads(result.output)["passed"] is False


@patch("aws_bootstrap.cli.cluster_mod.run_canary")
@patch("aws_bootstrap.cli.run_on_host", return_value=(0, "", ""))
@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_prepare_verifies_and_runs_canary(mock_session, mock_find, mock_gpu, mock_run, mock_canary, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_gpu.return_value = GpuInfo(
        driver_version="550",
        cuda_driver_version="12.4",
        cuda_toolkit_version="12.4",
        gpu_name="T4",
        compute_capability="7.5",
        architecture="Turing",
    )
    mock_canary.return_value = [NodeResult("i-0", 0, 0, "ok", ""), NodeResult("i-1", 1, 0, "ok", "")]
    result = _runner().invoke(
        main, ["cluster", "prepare", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))]
    )
    assert result.exit_code == 0, result.output
    assert mock_canary.called
    assert mock_run.call_count >= 2  # wrote per-node config


@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_prepare_fails_on_version_skew(mock_session, mock_find, mock_gpu, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_gpu.side_effect = [
        GpuInfo("550", "12.4", "12.4", "T4", "7.5", "Turing"),
        GpuInfo("550", "12.1", "12.1", "T4", "7.5", "Turing"),
    ]
    result = _runner().invoke(
        main, ["cluster", "prepare", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))]
    )
    assert result.exit_code != 0
    assert "mismatch" in result.output.lower()


# ---------------------------------------------------------------------------
# cluster run (Phase 3)
# ---------------------------------------------------------------------------


@patch("aws_bootstrap.cli.cluster_mod.run_distributed_job")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_run_distributes_and_reports(mock_session, mock_find, mock_run, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_run.return_value = [
        NodeResult("i-0", 0, 0, "loss 0.1", ""),
        NodeResult("i-1", 1, 0, "loss 0.1", ""),
    ]
    train = tmp_path / "train.py"
    train.write_text("print('hi')\n")
    log_dir = tmp_path / "logs"
    result = _runner().invoke(
        main,
        [
            "-o",
            "json",
            "cluster",
            "run",
            "--cluster-id",
            "ml1",
            "--key-path",
            str(_make_key(tmp_path)),
            "--log-dir",
            str(log_dir),
            str(train),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["succeeded"] is True
    mock_run.assert_called_once()
    assert (log_dir / "ml1" / "rank0.log").exists()


@patch("aws_bootstrap.cli.cluster_mod.run_distributed_job")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_run_nonzero_on_failure(mock_session, mock_find, mock_run, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0)]
    mock_run.return_value = [NodeResult("i-0", 0, 1, "", "traceback")]
    train = tmp_path / "train.py"
    train.write_text("x\n")
    result = _runner().invoke(
        main,
        [
            "-o",
            "json",
            "cluster",
            "run",
            "--cluster-id",
            "ml1",
            "--key-path",
            str(_make_key(tmp_path)),
            "--log-dir",
            str(tmp_path / "logs"),
            str(train),
        ],
    )
    assert result.exit_code != 0
    assert json.loads(result.output)["succeeded"] is False


def test_cluster_run_missing_script_errors(tmp_path):
    with patch("aws_bootstrap.cli.boto3.Session"):
        result = _runner().invoke(
            main,
            [
                "cluster",
                "run",
                "--cluster-id",
                "ml1",
                "--key-path",
                str(_make_key(tmp_path)),
                str(tmp_path / "nope.py"),
            ],
        )
    assert result.exit_code != 0
