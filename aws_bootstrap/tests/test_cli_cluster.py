"""CliRunner tests for the `cluster` command group (launch/status/terminate, prepare/test, run).

``_node`` is kept local because it encodes the cluster-specific dict
``find_cluster_instances`` returns.
"""

from __future__ import annotations
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aws_bootstrap.cli import main
from aws_bootstrap.cluster import NodeResult
from aws_bootstrap.ec2 import CLIError
from aws_bootstrap.gpu import GpuInfo
from aws_bootstrap.iam import InstanceProfileError


def _node(instance_id="i-1", cluster_id="ml1", rank=0, state="running", **overrides):
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
        "IamInstanceProfile": "",
        **overrides,
    }


@pytest.fixture
def cluster_nodes(cli_session):
    """Patch ``find_cluster_instances`` (no nodes by default); yields the mock."""
    with patch("aws_bootstrap.cli.find_cluster_instances", return_value=[]) as find:
        yield find


@pytest.fixture
def train_script(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print('hi')\n")
    return script


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["cluster", "--help"], ["launch", "status", "prepare", "test", "run", "terminate"]),
        (["cluster", "launch", "--help"], ["--wait", "--wait-timeout"]),
    ],
    ids=["group", "launch"],
)
def test_cluster_help(runner, argv, expected):
    result = runner.invoke(main, argv)
    assert result.exit_code == 0
    for text in expected:
        assert text in result.output


# ---------------------------------------------------------------------------
# status / terminate
# ---------------------------------------------------------------------------


def test_cluster_status_structured_output_lists_nodes(runner, cluster_nodes, structured):
    fmt, load = structured
    cluster_nodes.return_value = [_node(instance_id="i-aaa", rank=0)]
    result = runner.invoke(main, ["-o", fmt, "cluster", "status", "--cluster-id", "ml1"])
    assert result.exit_code == 0, result.output
    node = load(result.output)["nodes"][0]
    assert (node["rank"], node["instance_id"]) == (0, "i-aaa")


def test_cluster_status_table_has_real_columns(runner, cluster_nodes):
    cluster_nodes.return_value = [_node(instance_id="i-aaa", rank=0)]
    result = runner.invoke(main, ["-o", "table", "cluster", "status", "--cluster-id", "ml1"])
    assert result.exit_code == 0, result.output
    assert "i-aaa" in result.output
    assert "Instance" in result.output  # column header, not a key-value dump
    assert "'instance_id'" not in result.output  # not a Python repr of the node list


def test_cluster_status_still_shows_non_running_nodes(runner, cluster_nodes):
    """status is the diagnostic view — it must keep reporting a stopped node."""
    cluster_nodes.return_value = [_node("i-1", rank=0, state="running"), _node("i-2", rank=1, state="stopped")]
    result = runner.invoke(main, ["-o", "json", "cluster", "status", "--cluster-id", "ml1", "--region", "us-east-1"])
    assert result.exit_code == 0
    states = {n["rank"]: n["state"] for n in json.loads(result.output)["nodes"]}
    assert states == {0: "running", 1: "stopped"}


def test_cluster_terminate_requires_yes_in_json_mode(runner, cluster_nodes):
    cluster_nodes.return_value = [_node()]
    result = runner.invoke(main, ["-o", "json", "cluster", "terminate", "--cluster-id", "ml1"])
    assert result.exit_code != 0


@pytest.fixture
def cluster_terminate(cluster_nodes):
    cluster_nodes.return_value = [_node()]
    with (
        patch("aws_bootstrap.cli.terminate_tagged_instances", return_value=[]) as terminate,
        patch("aws_bootstrap.cli.remove_ssh_host", return_value="aws-ml1-0"),
        patch("aws_bootstrap.cli.delete_cluster_placement_group", return_value=True) as delete_pg,
    ):
        yield SimpleNamespace(terminate=terminate, delete_pg=delete_pg)


def test_cluster_terminate_waits_then_deletes_group(runner, cli_session, cluster_terminate):
    result = runner.invoke(main, ["cluster", "terminate", "--cluster-id", "ml1", "--yes"])
    assert result.exit_code == 0, result.output
    cluster_terminate.terminate.assert_called_once()
    assert cluster_terminate.terminate.call_args[0][1] == ["i-1"]
    # Waits for instances to fully terminate before deleting the placement group.
    cli_session.return_value.client.return_value.get_waiter.assert_called_with("instance_terminated")
    cluster_terminate.delete_pg.assert_called_once()


def test_cluster_terminate_warns_when_group_still_in_use(runner, cluster_terminate):
    cluster_terminate.delete_pg.return_value = False
    result = runner.invoke(main, ["cluster", "terminate", "--cluster-id", "ml1", "--yes"])
    # Instances still terminate; the in-use placement group is a warning, not a crash.
    assert result.exit_code == 0, result.output
    assert "still in use" in result.output.lower()


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_launch(launched, region_launch, cluster_nodes, key_path):
    """Patch a cluster launch in us-east-1c; yields the ``launched`` mocks and the base argv."""
    launched.launch.side_effect = lambda config, prepare_region, **kw: region_launch(
        {"InstanceId": "i-x"},
        region="us-east-1",
        placement_azs=("us-east-1c",),
        placement_group="aws-bootstrap-cluster-ml1",
    )
    launched.ready.return_value = {"PublicIpAddress": "1.2.3.4", "Placement": {"AvailabilityZone": "us-east-1c"}}
    launched.add_ssh.return_value = "aws-ml1-0"
    with (
        patch("aws_bootstrap.cli.ensure_cluster_placement_group", return_value="aws-bootstrap-cluster-ml1") as pg,
        patch("aws_bootstrap.cli.ensure_cluster_security_group_rule") as sg_rule,
    ):
        argv = ["cluster", "launch", "--cluster-id", "ml1", "--region", "us-east-1", "--key-path", str(key_path)]
        yield SimpleNamespace(mocks=launched, argv=argv, nodes=cluster_nodes, pg=pg, sg_rule=sg_rule)


def test_cluster_launch_two_nodes_tags_ranks(runner, cluster_launch):
    result = runner.invoke(main, [*cluster_launch.argv, "--nodes", "2", "--no-setup"])
    assert result.exit_code == 0, result.output
    launch = cluster_launch.mocks.launch
    assert launch.call_count == 2
    aliases = {c.kwargs.get("alias") for c in cluster_launch.mocks.add_ssh.call_args_list}
    assert aliases == {"aws-ml1-0", "aws-ml1-1"}
    # Cluster launch must NOT silently fall back to on-demand per node: it passes
    # a confirm_on_demand callback that always declines.
    confirm = launch.call_args.kwargs.get("confirm_on_demand")
    assert confirm is not None and confirm() is False
    # --no-setup: no node waits for SSH or runs remote setup.
    cluster_launch.mocks.ssh_ready.assert_not_called()
    cluster_launch.mocks.setup.assert_not_called()


def test_cluster_launch_runs_remote_setup_per_node(runner, cluster_launch):
    """Regression: cluster launch MUST run remote setup (so nodes get ~/venv +
    torchrun) — without it prepare/test/run fail on real hardware."""
    result = runner.invoke(main, [*cluster_launch.argv, "--nodes", "2"])
    assert result.exit_code == 0, result.output
    assert cluster_launch.mocks.setup.call_count == 2  # remote setup ran on every node


def test_cluster_launch_threads_wait_into_config(runner, cluster_launch):
    args = ["--nodes", "1", "--no-setup", "--wait", "--wait-timeout", "90s"]
    result = runner.invoke(main, [*cluster_launch.argv, *args])
    assert result.exit_code == 0, result.output
    # The LaunchConfig handed to launch_with_retry carries the wait settings.
    cfg = cluster_launch.mocks.launch.call_args[0][0]
    assert cfg.wait is True
    assert cfg.wait_timeout == 90


PROFILE = ["--iam-instance-profile", "app"]


def test_cluster_launch_checks_and_threads_the_profile(runner, cluster_launch):
    result = runner.invoke(main, ["-o", "json", *cluster_launch.argv, "--nodes", "1", "--no-setup", *PROFILE])
    assert result.exit_code == 0, result.output
    cluster_launch.mocks.profile_check.assert_called_once()
    assert cluster_launch.mocks.launch.call_args[0][0].iam_instance_profile == "app"
    assert json.loads(result.stdout)["iam_instance_profile"] == "app"
    assert "WARNING" not in result.output  # a new cluster has no nodes to disagree with


def test_cluster_launch_profile_check_fails_before_anything_is_created(runner, cluster_launch):
    cluster_launch.mocks.profile_check.side_effect = CLIError("IAM instance profile 'app' not found.")
    result = runner.invoke(main, [*cluster_launch.argv, "--nodes", "1", *PROFILE])
    assert result.exit_code != 0
    assert "'app' not found" in result.output
    cluster_launch.pg.assert_not_called()
    cluster_launch.mocks.sg.assert_not_called()
    cluster_launch.sg_rule.assert_not_called()
    cluster_launch.mocks.launch.assert_not_called()


@pytest.mark.parametrize(
    "existing_profile,flag,expected,hint,no_hint",
    [
        pytest.param(
            "", PROFILE, "profile none; new nodes get app", "associate-iam-instance-profile", "replace-", id="add"
        ),
        pytest.param(
            "old", PROFILE, "profile old; new nodes get app", "replace-iam-instance-profile", "associate-", id="swap"
        ),
        pytest.param("old", [], "profile old; new nodes get none", None, "aws ec2", id="drop"),
    ],
)
def test_growing_a_cluster_warns_about_mismatched_profiles(
    runner, cluster_launch, existing_profile, flag, expected, hint, no_hint
):
    cluster_launch.nodes.return_value = [_node("i-0", IamInstanceProfile=existing_profile)]
    result = runner.invoke(main, [*cluster_launch.argv, "--nodes", "2", "--no-setup", *flag])
    assert result.exit_code == 0, result.output
    assert f"node(s) i-0 have IAM instance {expected}" in result.output
    assert hint is None or hint in result.output
    assert no_hint not in result.output


def test_cluster_profile_failure_keeps_the_profile_in_the_rerun_hint_and_skips_capacity_advice(
    runner, cluster_launch, region_launch
):
    calls = iter([region_launch({"InstanceId": "i-x"}, region="us-east-1")])

    def launch(config, prepare_region, **kw):
        try:
            return next(calls)
        except StopIteration:
            raise InstanceProfileError("EC2 rejected IAM instance profile 'app': not yet propagated") from None

    cluster_launch.mocks.launch.side_effect = launch
    result = runner.invoke(main, [*cluster_launch.argv, "--nodes", "2", "--no-setup", *PROFILE])
    assert result.exit_code != 0
    assert "Launched 1 of 2" in result.output
    assert "--nodes 2 --region us-east-1 --iam-instance-profile app" in result.output
    assert "EC2 rejected IAM instance profile 'app'" in result.output
    assert "ride out spot capacity" not in result.output


def test_cluster_launch_reports_nothing_to_add_for_a_half_dead_cluster(runner, cluster_nodes):
    """Pins the behaviour the remediation wording has to account for."""
    cluster_nodes.return_value = [_node("i-1", rank=0, state="running"), _node("i-2", rank=1, state="shutting-down")]
    result = runner.invoke(main, ["cluster", "launch", "--cluster-id", "ml1", "--nodes", "2", "--region", "us-east-1"])
    assert "nothing to add" in result.output


# ---------------------------------------------------------------------------
# prepare / test
# ---------------------------------------------------------------------------

_T4_CUDA_12_4 = GpuInfo("550", "12.4", "12.4", "T4", "7.5", "Turing")


@pytest.mark.parametrize("rc,passed", [(0, True), (1, False)], ids=["pass", "fail"])
def test_cluster_test_reports_canary_outcome(runner, cluster_nodes, key_path, rc, passed):
    cluster_nodes.return_value = [_node(instance_id="i-0", rank=0)]
    outcome = [NodeResult("i-0", 0, rc, "[canary] out", "boom" if rc else "")]
    argv = ["-o", "json", "cluster", "test", "--cluster-id", "ml1", "--key-path", str(key_path)]
    with patch("aws_bootstrap.cli.cluster_mod.run_canary", return_value=outcome) as mock_canary:
        result = runner.invoke(main, argv)
    assert json.loads(result.output)["passed"] is passed
    assert (result.exit_code == 0) is passed
    mock_canary.assert_called_once()


def test_cluster_prepare_verifies_and_runs_canary(runner, cluster_nodes, key_path):
    cluster_nodes.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    outcome = [NodeResult("i-0", 0, 0, "ok", ""), NodeResult("i-1", 1, 0, "ok", "")]
    with (
        patch("aws_bootstrap.cli.query_gpu_info", return_value=_T4_CUDA_12_4),
        patch("aws_bootstrap.cli.run_on_host", return_value=(0, "", "")) as mock_run,
        patch("aws_bootstrap.cli.cluster_mod.run_canary", return_value=outcome) as mock_canary,
    ):
        result = runner.invoke(main, ["cluster", "prepare", "--cluster-id", "ml1", "--key-path", str(key_path)])
    assert result.exit_code == 0, result.output
    assert mock_canary.called
    assert mock_run.call_count >= 2  # wrote per-node config


def test_cluster_prepare_fails_on_version_skew(runner, cluster_nodes, key_path):
    cluster_nodes.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    skewed = [_T4_CUDA_12_4, GpuInfo("550", "12.1", "12.1", "T4", "7.5", "Turing")]
    with patch("aws_bootstrap.cli.query_gpu_info", side_effect=skewed):
        result = runner.invoke(main, ["cluster", "prepare", "--cluster-id", "ml1", "--key-path", str(key_path)])
    assert result.exit_code != 0
    assert "mismatch" in result.output.lower()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rc,succeeded", [(0, True), (1, False)], ids=["success", "failure"])
def test_cluster_run_reports_outcome(runner, cluster_nodes, key_path, train_script, tmp_path, rc, succeeded):
    cluster_nodes.return_value = [_node(instance_id="i-0", rank=0)]
    log_dir = tmp_path / "logs"
    outcome = [NodeResult("i-0", 0, rc, "loss 0.1", "traceback" if rc else "")]
    argv = ["cluster", "run", "--cluster-id", "ml1", "--key-path", str(key_path), "--log-dir", str(log_dir)]
    with patch("aws_bootstrap.cli.cluster_mod.run_distributed_job", return_value=outcome) as mock_run:
        result = runner.invoke(main, ["-o", "json", *argv, str(train_script)])
    assert json.loads(result.output)["succeeded"] is succeeded
    assert (result.exit_code == 0) is succeeded
    mock_run.assert_called_once()
    # Per-node logs are written regardless of outcome.
    assert (log_dir / "ml1" / "rank0.log").exists()


def test_cluster_run_echoes_output_from_any_node(runner, cluster_nodes, key_path, train_script, tmp_path):
    # Under c10d rendezvous, torch global rank 0 (which prints) can land on a
    # node that is NOT our node rank 0 — so output from any node must surface.
    cluster_nodes.return_value = [_node(instance_id="i-0", rank=0), _node(instance_id="i-1", rank=1)]
    outcome = [
        NodeResult("i-0", 0, 0, "", ""),  # node rank 0 produced no stdout
        NodeResult("i-1", 1, 0, "DONE world_size=2", ""),  # the printing rank
    ]
    argv = ["cluster", "run", "--cluster-id", "ml1", "--key-path", str(key_path), "--log-dir", str(tmp_path / "logs")]
    with patch("aws_bootstrap.cli.cluster_mod.run_distributed_job", return_value=outcome):
        result = runner.invoke(main, [*argv, str(train_script)])
    assert result.exit_code == 0, result.output
    assert "DONE world_size=2" in result.output  # surfaced even though it was rank 1


def test_cluster_run_missing_script_errors(runner, cli_session, key_path, tmp_path):
    argv = ["cluster", "run", "--cluster-id", "ml1", "--key-path", str(key_path), str(tmp_path / "nope.py")]
    result = runner.invoke(main, argv)
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Execution commands require every node to be running
#
# `find_cluster_instances` deliberately reports non-terminated nodes so `status`
# can show a stopped one. prepare/test/run must not: a spot-reclaimed rank 0 is
# still what `master_addr` picks, and every node then hangs waiting to
# rendezvous against an IP that is going away.
# ---------------------------------------------------------------------------


_EXEC_COMMANDS = [
    ["cluster", "prepare", "--cluster-id", "ml1", "--region", "us-east-1"],
    ["cluster", "test", "--cluster-id", "ml1", "--region", "us-east-1"],
    ["cluster", "run", "--cluster-id", "ml1", "--region", "us-east-1", "train.py"],
]


@pytest.mark.parametrize(
    "nodes,expected",
    [
        pytest.param(
            [_node("i-1", rank=0, state="running"), _node("i-2", rank=1, state="shutting-down")],
            # The remediation has to be actionable, not just a complaint.
            ["not running", "rank 1 (shutting-down)", "cluster launch", "cluster terminate"],
            id="partially-running",
        ),
        pytest.param(
            [_node("i-1", rank=0, state="stopped"), _node("i-2", rank=1, state="stopped")],
            ["No running nodes", "rank 0=stopped"],
            id="fully-stopped",
        ),
        pytest.param([], ["No nodes found for cluster 'ml1'"], id="unknown-cluster"),
    ],
)
@pytest.mark.parametrize("argv", _EXEC_COMMANDS, ids=["prepare", "test", "run"])
def test_cluster_exec_refuses_unless_every_node_is_running(runner, cluster_nodes, train_script, argv, nodes, expected):
    cluster_nodes.return_value = nodes
    result = runner.invoke(main, [a.replace("train.py", str(train_script)) for a in argv])
    assert result.exit_code != 0
    for text in expected:
        assert text in result.output


def test_partial_cluster_remediation_leads_with_terminate_not_launch(runner, cluster_nodes):
    """`cluster launch --nodes <total>` is a no-op here: it counts every
    non-terminated node, so it sees the cluster as already full and prints
    'nothing to add'. The remediation must not send the user down that path."""
    cluster_nodes.return_value = [_node("i-1", rank=0, state="running"), _node("i-2", rank=1, state="shutting-down")]
    result = runner.invoke(main, ["cluster", "test", "--cluster-id", "ml1", "--region", "us-east-1"])
    assert result.exit_code != 0
    out = result.output
    assert "cluster terminate" in out
    assert out.index("cluster terminate") < out.index("cluster launch")
    assert "will not help" in out
