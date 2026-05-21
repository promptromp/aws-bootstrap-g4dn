"""CliRunner tests for the `cluster` command group (Phases 1-3).

Uses the shared ``runner`` fixture from ``conftest.py``. ``_node`` / ``_make_key``
are kept local because they encode cluster-specific shapes (the dict
``find_cluster_instances`` returns, and a throwaway public key).
"""

from __future__ import annotations
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from aws_bootstrap.cli import main
from aws_bootstrap.cluster import NodeResult
from aws_bootstrap.ec2 import RegionContext, RegionLaunch
from aws_bootstrap.gpu import GpuInfo


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


# ---------------------------------------------------------------------------
# launch / status / terminate (Phase 1)
# ---------------------------------------------------------------------------


def test_cluster_help_lists_subcommands(runner):
    result = runner.invoke(main, ["cluster", "--help"])
    assert result.exit_code == 0
    for sub in ("launch", "status", "prepare", "test", "run", "terminate"):
        assert sub in result.output


def test_cluster_status_json_lists_nodes(runner):
    with (
        patch("aws_bootstrap.cli.boto3.Session"),
        patch("aws_bootstrap.cli.find_cluster_instances", return_value=[_node()]),
    ):
        result = runner.invoke(main, ["-o", "json", "cluster", "status", "--cluster-id", "ml1"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["nodes"][0]["rank"] == 0
    assert data["nodes"][0]["instance_id"] == "i-1"


def test_cluster_terminate_requires_yes_in_json_mode(runner):
    with (
        patch("aws_bootstrap.cli.boto3.Session"),
        patch("aws_bootstrap.cli.find_cluster_instances", return_value=[_node()]),
    ):
        result = runner.invoke(main, ["-o", "json", "cluster", "terminate", "--cluster-id", "ml1"])
    assert result.exit_code != 0


@patch("aws_bootstrap.cli.delete_cluster_placement_group", return_value=True)
@patch("aws_bootstrap.cli.remove_ssh_host", return_value="aws-ml1-0")
@patch("aws_bootstrap.cli.terminate_tagged_instances", return_value=[])
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_terminate_waits_then_deletes_group(
    mock_session, mock_find, mock_term, mock_remove, mock_delpg, runner
):
    mock_find.return_value = [_node()]
    ec2 = mock_session.return_value.client.return_value
    result = runner.invoke(main, ["cluster", "terminate", "--cluster-id", "ml1", "--yes"])
    assert result.exit_code == 0, result.output
    mock_term.assert_called_once()
    assert mock_term.call_args[0][1] == ["i-1"]
    # Waits for instances to fully terminate before deleting the placement group.
    ec2.get_waiter.assert_called_with("instance_terminated")
    mock_delpg.assert_called_once()


@patch("aws_bootstrap.cli.delete_cluster_placement_group", return_value=False)
@patch("aws_bootstrap.cli.remove_ssh_host", return_value="aws-ml1-0")
@patch("aws_bootstrap.cli.terminate_tagged_instances", return_value=[])
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_terminate_warns_when_group_still_in_use(
    mock_session, mock_find, mock_term, mock_remove, mock_delpg, runner
):
    mock_find.return_value = [_node()]
    result = runner.invoke(main, ["cluster", "terminate", "--cluster-id", "ml1", "--yes"])
    # Instances still terminate; the in-use placement group is a warning, not a crash.
    assert result.exit_code == 0, result.output
    assert "still in use" in result.output.lower()


def _launch_side_effect(config, prepare_region, **kw):
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
    runner,
    tmp_path,
):
    mock_launch.side_effect = _launch_side_effect
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-east-1c"}}

    result = runner.invoke(
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
            str(_make_key(tmp_path)),
            "--no-setup",
        ],
    )
    assert result.exit_code == 0, result.output
    assert mock_launch.call_count == 2
    aliases = {c.kwargs.get("alias") for c in mock_add_ssh.call_args_list}
    assert aliases == {"aws-ml1-0", "aws-ml1-1"}
    # Cluster launch must NOT silently fall back to on-demand per node: it passes
    # a confirm_on_demand callback that always declines.
    confirm = mock_launch.call_args.kwargs.get("confirm_on_demand")
    assert confirm is not None and confirm() is False


@patch("aws_bootstrap.cli.run_remote_setup", return_value=True)
@patch("aws_bootstrap.cli.wait_for_ssh", return_value=True)
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
def test_cluster_launch_runs_remote_setup_per_node(
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
    mock_wait_ssh,
    mock_setup,
    runner,
    tmp_path,
):
    """Regression: cluster launch MUST run remote setup (so nodes get ~/venv +
    torchrun) — without it prepare/test/run fail on real hardware."""
    mock_launch.side_effect = _launch_side_effect
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-east-1c"}}

    result = runner.invoke(
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
            str(_make_key(tmp_path)),
        ],
    )
    assert result.exit_code == 0, result.output
    assert mock_setup.call_count == 2  # remote setup ran on every node


def test_cluster_launch_help_shows_wait(runner):
    result = runner.invoke(main, ["cluster", "launch", "--help"])
    assert result.exit_code == 0
    assert "--wait" in result.output
    assert "--wait-timeout" in result.output


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
def test_cluster_launch_threads_wait_into_config(
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
    runner,
    tmp_path,
):
    mock_launch.side_effect = _launch_side_effect
    mock_wait.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-east-1c"}}
    result = runner.invoke(
        main,
        [
            "cluster",
            "launch",
            "--cluster-id",
            "ml1",
            "--nodes",
            "1",
            "--region",
            "us-east-1",
            "--key-path",
            str(_make_key(tmp_path)),
            "--no-setup",
            "--wait",
            "--wait-timeout",
            "90s",
        ],
    )
    assert result.exit_code == 0, result.output
    # The LaunchConfig handed to launch_with_retry carries the wait settings.
    cfg = mock_launch.call_args[0][0]
    assert cfg.wait is True
    assert cfg.wait_timeout == 90


# ---------------------------------------------------------------------------
# prepare / test (Phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rc,passed,exits_nonzero", [(0, True, False), (1, False, True)])
@patch("aws_bootstrap.cli.cluster_mod.run_canary")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_test_reports_canary_outcome(
    mock_session, mock_find, mock_canary, runner, tmp_path, rc, passed, exits_nonzero
):
    mock_find.return_value = [_node(instance_id="i-0", rank=0)]
    mock_canary.return_value = [NodeResult("i-0", 0, rc, "[canary] out", "boom" if rc else "")]
    result = runner.invoke(
        main, ["-o", "json", "cluster", "test", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))]
    )
    data = json.loads(result.output)
    assert data["passed"] is passed
    assert (result.exit_code != 0) is exits_nonzero
    mock_canary.assert_called_once()


@patch("aws_bootstrap.cli.cluster_mod.run_canary")
@patch("aws_bootstrap.cli.run_on_host", return_value=(0, "", ""))
@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_prepare_verifies_and_runs_canary(
    mock_session, mock_find, mock_gpu, mock_run, mock_canary, runner, tmp_path
):
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_gpu.return_value = GpuInfo("550", "12.4", "12.4", "T4", "7.5", "Turing")
    mock_canary.return_value = [NodeResult("i-0", 0, 0, "ok", ""), NodeResult("i-1", 1, 0, "ok", "")]
    result = runner.invoke(main, ["cluster", "prepare", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))])
    assert result.exit_code == 0, result.output
    assert mock_canary.called
    assert mock_run.call_count >= 2  # wrote per-node config


@patch("aws_bootstrap.cli.query_gpu_info")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_prepare_fails_on_version_skew(mock_session, mock_find, mock_gpu, runner, tmp_path):
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_gpu.side_effect = [
        GpuInfo("550", "12.4", "12.4", "T4", "7.5", "Turing"),
        GpuInfo("550", "12.1", "12.1", "T4", "7.5", "Turing"),
    ]
    result = runner.invoke(main, ["cluster", "prepare", "--cluster-id", "ml1", "--key-path", str(_make_key(tmp_path))])
    assert result.exit_code != 0
    assert "mismatch" in result.output.lower()


# ---------------------------------------------------------------------------
# run (Phase 3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rc,succeeded,exits_nonzero", [(0, True, False), (1, False, True)])
@patch("aws_bootstrap.cli.cluster_mod.run_distributed_job")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_run_reports_outcome(mock_session, mock_find, mock_run, runner, tmp_path, rc, succeeded, exits_nonzero):
    mock_find.return_value = [_node(instance_id="i-0", rank=0)]
    mock_run.return_value = [NodeResult("i-0", 0, rc, "loss 0.1", "traceback" if rc else "")]
    train = tmp_path / "train.py"
    train.write_text("print('hi')\n")
    log_dir = tmp_path / "logs"
    result = runner.invoke(
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
    data = json.loads(result.output)
    assert data["succeeded"] is succeeded
    assert (result.exit_code != 0) is exits_nonzero
    mock_run.assert_called_once()
    # Per-node logs are written regardless of outcome.
    assert (log_dir / "ml1" / "rank0.log").exists()


@patch("aws_bootstrap.cli.cluster_mod.run_distributed_job")
@patch("aws_bootstrap.cli.find_cluster_instances")
@patch("aws_bootstrap.cli.boto3.Session")
def test_cluster_run_echoes_output_from_any_node(mock_session, mock_find, mock_run, runner, tmp_path):
    # Under c10d rendezvous, torch global rank 0 (which prints) can land on a
    # node that is NOT our node rank 0 — so output from any node must surface.
    mock_find.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    mock_run.return_value = [
        NodeResult("i-0", 0, 0, "", ""),  # node rank 0 produced no stdout
        NodeResult("i-1", 1, 0, "DONE world_size=2", ""),  # the printing rank
    ]
    train = tmp_path / "train.py"
    train.write_text("x\n")
    result = runner.invoke(
        main,
        [
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
    assert result.exit_code == 0, result.output
    assert "DONE world_size=2" in result.output  # surfaced even though it was rank 1


def test_cluster_run_missing_script_errors(runner, tmp_path):
    with patch("aws_bootstrap.cli.boto3.Session"):
        result = runner.invoke(
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
