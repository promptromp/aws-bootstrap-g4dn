"""Tests for cluster composition helpers (aws_bootstrap.cluster)."""

from __future__ import annotations
import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aws_bootstrap import cluster
from aws_bootstrap.cluster import master_addr
from aws_bootstrap.config import LaunchConfig
from aws_bootstrap.ec2 import RegionContext, RegionLaunch


@pytest.mark.parametrize(
    "cluster_id,expected",
    [("ml1", "aws-bootstrap-cluster-ml1"), ("Exp_2", "aws-bootstrap-cluster-Exp_2")],
)
def test_placement_group_name(cluster_id, expected):
    assert cluster.placement_group_name(cluster_id) == expected


@pytest.mark.parametrize(
    "cluster_id,rank,expected",
    [("ml1", 0, "aws-ml1-0"), ("ml1", 3, "aws-ml1-3")],
)
def test_node_alias(cluster_id, rank, expected):
    assert cluster.node_alias(cluster_id, rank) == expected


@pytest.mark.parametrize(
    "current,target,expected",
    [(0, 4, 4), (2, 4, 2), (4, 4, 0), (5, 4, 0)],
)
def test_nodes_to_add(current, target, expected):
    assert cluster.nodes_to_add(current, target) == expected


# ---------------------------------------------------------------------------
# Launch fan-out
# ---------------------------------------------------------------------------


def _fake_launch(rank_holder):
    """A launch_fn that returns a distinct RegionLaunch per call."""

    def launch_fn(config, prepare_region, **kwargs):
        i = rank_holder["n"]
        rank_holder["n"] += 1
        ctx = RegionContext(
            region="us-east-1",
            ec2_client=MagicMock(),
            ami={"ImageId": "ami-x"},
            sg_id="sg-1",
            key_name="k",
            placement_azs=("us-east-1c",),
            placement_group="aws-bootstrap-cluster-ml1",
        )
        return RegionLaunch("us-east-1", ctx, {"InstanceId": f"i-{i}"}, "spot")

    return launch_fn


def test_launch_cluster_nodes_assigns_sequential_ranks_and_reports_each_node():
    seen = []
    results = cluster.launch_cluster_nodes(
        LaunchConfig(regions=("us-east-1",)),
        prepare_region=lambda r: None,
        count=3,
        start_rank=0,
        launch_fn=_fake_launch({"n": 0}),
        on_node=lambda rank, launch: seen.append((rank, launch.instance["InstanceId"])),
    )
    assert [r.rank for r in results] == [0, 1, 2]
    assert [r.launch.instance["InstanceId"] for r in results] == ["i-0", "i-1", "i-2"]
    assert seen == [(0, "i-0"), (1, "i-1"), (2, "i-2")]


def test_launch_cluster_nodes_starts_at_offset_without_a_callback():
    """An incremental launch continues the existing ranks; on_node is optional."""
    results = cluster.launch_cluster_nodes(
        LaunchConfig(regions=("us-east-1",)),
        prepare_region=lambda r: None,
        count=2,
        start_rank=2,
        launch_fn=_fake_launch({"n": 0}),
    )
    assert [r.rank for r in results] == [2, 3]
    assert [r.launch.instance["InstanceId"] for r in results] == ["i-0", "i-1"]


# ---------------------------------------------------------------------------
# Pure orchestration helpers (master addr, torchrun command, env, version skew)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nodes",
    [
        pytest.param([{"Rank": 1, "PrivateIp": "10.0.0.6"}, {"Rank": 0, "PrivateIp": "10.0.0.5"}], id="ranked"),
        # A node whose rank tag failed to write (Rank=None) must not crash master_addr.
        pytest.param(
            [{"Rank": None, "PrivateIp": "10.0.0.9"}, {"Rank": 0, "PrivateIp": "10.0.0.5"}], id="unknown-rank"
        ),
    ],
)
def test_master_addr_is_rank0_private_ip(nodes):
    assert cluster.master_addr(nodes) == "10.0.0.5"


def test_build_torchrun_command_quotes_script_args():
    # Args with spaces/special chars must be shell-quoted so the remote shell
    # doesn't word-split them.
    cmd = cluster.build_torchrun_command(
        "/tmp/t.py", 1, 1, "10.0.0.5", "ml1", 29400, script_args=["--run-name", "my run", "--flag"]
    )
    assert "'my run'" in cmd
    assert "/tmp/t.py --run-name 'my run' --flag" in cmd


@pytest.mark.parametrize(
    "num_nodes,script,script_args,expected",
    [
        (
            2,
            "/tmp/t.py",
            None,
            "torchrun --nnodes=2 --nproc-per-node=1 "
            "--rdzv-backend=c10d --rdzv-endpoint=10.0.0.5:29400 --rdzv-id=ml1 /tmp/t.py",
        ),
        (
            4,
            "train.py",
            ["--epochs", "1"],
            "torchrun --nnodes=4 --nproc-per-node=1 "
            "--rdzv-backend=c10d --rdzv-endpoint=10.0.0.5:29400 --rdzv-id=ml1 train.py --epochs 1",
        ),
    ],
    ids=["no-args", "with-args"],
)
def test_build_torchrun_command_exact(num_nodes, script, script_args, expected):
    # Exact match: this string is the load-bearing contract with torchrun, so
    # guard against silent flag-format drift (e.g. --nproc-per-node vs _).
    cmd = cluster.build_torchrun_command(script, num_nodes, 1, "10.0.0.5", "ml1", 29400, script_args=script_args)
    assert cmd == expected


def test_render_node_config_is_sourceable():
    env = cluster.node_env("ml1", 0, 2, 1, ["10.0.0.5", "10.0.0.6"], "10.0.0.5")
    text = cluster.render_node_config(env)
    # Each assignment is an `export` line (shell-safe scalars left unquoted).
    assert "export AWSB_NODE_RANK=0" in text
    assert "export AWSB_MASTER_ADDR=10.0.0.5" in text
    # The multi-line node-IPs value stays inside ONE quoted assignment, so the
    # second IP is not a bare standalone line the shell would execute on source.
    assert "export AWSB_NODE_IPS='10.0.0.5\n10.0.0.6'" in text
    assert "\n10.0.0.6\n" not in f"\n{text}\n"


def test_node_env_contract():
    env = cluster.node_env(
        cluster_id="ml1",
        node_rank=2,
        num_nodes=4,
        num_gpus_per_node=1,
        node_ips=["10.0.0.5", "10.0.0.6"],
        master_addr="10.0.0.5",
    )
    assert env["AWSB_CLUSTER_ID"] == "ml1"
    assert env["AWSB_NODE_RANK"] == "2"
    assert env["AWSB_NUM_NODES"] == "4"
    assert env["AWSB_NUM_GPUS_PER_NODE"] == "1"
    assert env["AWSB_MASTER_ADDR"] == "10.0.0.5"
    assert env["AWSB_NODE_IPS"] == "10.0.0.5\n10.0.0.6"


@pytest.mark.parametrize(
    "versions,expected_ok",
    [
        ({"i-0": "12.4", "i-1": "12.4"}, True),
        ({"i-0": "12.4", "i-1": "12.1"}, False),
        ({"i-0": "12.4"}, True),
    ],
    ids=["consistent", "skewed", "single-node"],
)
def test_detect_version_skew(versions, expected_ok):
    mismatches = cluster.detect_version_skew(versions)
    assert (len(mismatches) == 0) == expected_ok


def test_canary_resource_is_valid_python():
    src = Path("aws_bootstrap/resources/cluster_canary.py").read_text()
    ast.parse(src)  # raises SyntaxError if invalid
    assert 'if __name__ == "__main__"' in src
    assert "init_process_group" in src
    assert "all_reduce" in src


# ---------------------------------------------------------------------------
# Parallel multi-node execution + canary orchestration
# ---------------------------------------------------------------------------

TWO_NODES = [
    {"Rank": 0, "InstanceId": "i-0", "PrivateIp": "10.0.0.5", "PublicIp": "1.1.1.1"},
    {"Rank": 1, "InstanceId": "i-1", "PrivateIp": "10.0.0.6", "PublicIp": "2.2.2.2"},
]


class _Recorder:
    """A run_fn that records each command and answers ``outcome(node, command)``."""

    def __init__(self, outcome=lambda node, command: (0, "", "")):
        self.commands: list[str] = []
        self._outcome = outcome

    def __call__(self, node, command):
        self.commands.append(command)
        return self._outcome(node, command)


def test_run_on_all_nodes_collects_results_in_order():
    results = cluster.run_on_all_nodes(
        TWO_NODES, lambda node: "echo hi", run_fn=lambda node, cmd: (0, f"out-{node['InstanceId']}", "")
    )
    assert [r.instance_id for r in results] == ["i-0", "i-1"]
    assert [r.returncode for r in results] == [0, 0]
    assert results[0].stdout == "out-i-0"


def test_run_on_all_nodes_reports_failures():
    def run_fn(node, command):
        return (node["Rank"], "", "boom" if node["Rank"] else "")

    results = cluster.run_on_all_nodes(TWO_NODES, lambda node: "x", run_fn=run_fn)
    assert [r.returncode for r in results] == [0, 1]
    assert results[1].stderr == "boom"  # the node's own stderr, untouched


def test_run_on_all_nodes_survives_raising_run_fn():
    def run_fn(node, command):
        if node["Rank"] == 1:
            raise RuntimeError("kaboom")
        return (0, "ok", "")

    results = cluster.run_on_all_nodes(TWO_NODES, lambda node: "x", run_fn=run_fn)
    # One node raising must not abort the whole sweep; it's a rank-labeled failure.
    assert [r.returncode for r in results] == [0, 1]
    assert "kaboom" in results[1].stderr
    assert results[1].instance_id == "i-1"


def test_run_canary_scps_and_runs_torchrun_on_all_nodes():
    scped: list[str] = []
    run_fn = _Recorder(lambda node, cmd: (0, f"[canary] rank ok {node['Rank']}", ""))
    results = cluster.run_canary(
        TWO_NODES,
        cluster_id="ml1",
        nproc_per_node=1,
        rdzv_port=29400,
        scp_fn=lambda node, local, remote: scped.append(node["InstanceId"]) or True,
        run_fn=run_fn,
    )
    assert scped == ["i-0", "i-1"]
    assert len(run_fn.commands) == len(results) == 2
    assert all("--rdzv-endpoint=10.0.0.5:29400" in c for c in run_fn.commands)
    assert all("cluster_canary.py" in c for c in run_fn.commands)
    assert all(r.returncode == 0 for r in results)


def test_run_canary_scp_failure_aborts():
    run_fn = _Recorder()
    results = cluster.run_canary(
        TWO_NODES[:1],
        cluster_id="ml1",
        nproc_per_node=1,
        rdzv_port=29400,
        scp_fn=lambda node, local, remote: False,
        run_fn=run_fn,
    )
    assert run_fn.commands == []
    assert results[0].returncode != 0


# ---------------------------------------------------------------------------
# run_distributed_job (general runner; run_canary delegates to it)
# ---------------------------------------------------------------------------


def _run_job(run_fn, script="train.py", script_args=None, **kwargs):
    return cluster.run_distributed_job(
        TWO_NODES,
        cluster_id="ml1",
        nproc_per_node=1,
        rdzv_port=29400,
        local_script=Path(script),
        remote_script=f"/tmp/{script}",
        script_args=script_args,
        scp_fn=lambda n, ll, r: True,
        run_fn=run_fn,
        **kwargs,
    )


def test_run_distributed_job_py_uses_torchrun():
    run_fn = _Recorder(lambda node, cmd: (0, "done", ""))
    results = _run_job(run_fn, script_args=["--epochs", "2"])
    assert len(run_fn.commands) == len(results) == 2
    assert all(r.returncode == 0 for r in results)
    assert all("torchrun" in c and "/tmp/train.py --epochs 2" in c for c in run_fn.commands)
    assert all("--rdzv-endpoint=10.0.0.5:29400" in c for c in run_fn.commands)


def test_run_distributed_job_sh_uses_escape_hatch_env():
    run_fn = _Recorder()
    _run_job(run_fn, script="job.sh")
    ran = run_fn.commands
    assert len(ran) == 2
    assert all("bash /tmp/job.sh" in c for c in ran)
    assert all("AWSB_MASTER_ADDR=10.0.0.5" in c for c in ran)
    assert any("AWSB_NODE_RANK=0" in c for c in ran) and any("AWSB_NODE_RANK=1" in c for c in ran)
    assert all("torchrun" not in c for c in ran)


def test_run_distributed_job_runs_data_prep_before_training():
    run_fn = _Recorder()
    _run_job(run_fn, data_script=Path("prep.sh"))
    calls = run_fn.commands
    assert sum("data_prep.sh" in c for c in calls) == 2
    assert sum("torchrun" in c for c in calls) == 2
    # Training only after every node's data-prep finished.
    assert calls.index(next(c for c in calls if "torchrun" in c)) >= 2


def test_run_distributed_job_aborts_if_data_prep_fails():
    run_fn = _Recorder(lambda node, cmd: (1, "", "prep failed") if "data_prep.sh" in cmd else (0, "", ""))
    results = _run_job(run_fn, data_script=Path("prep.sh"))
    assert len(results) == 2
    assert all(r.returncode != 0 for r in results)
    assert len(run_fn.commands) == 2  # the two data-prep runs, and no training
    assert all("torchrun" not in c for c in run_fn.commands)


def test_master_addr_rejects_an_empty_node_list():
    """Better than the bare ValueError min() used to raise from inside a sort key."""
    with pytest.raises(ValueError, match="no cluster nodes"):
        master_addr([])
