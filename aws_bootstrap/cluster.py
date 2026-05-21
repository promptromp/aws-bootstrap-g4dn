"""Multi-node training cluster orchestration.

A cluster is a set of EC2 GPU instances sharing a ``--cluster-id`` tag,
launched into one AZ inside a cluster placement group so they can run a
distributed ``torchrun`` job. AWS tags are the source of truth — there is no
local cluster state file.

This module holds the composition logic (naming, rank/size math, launch
fan-out). AWS primitives live in :mod:`aws_bootstrap.ec2`; CLI wiring lives in
:mod:`aws_bootstrap.cli`.
"""

from __future__ import annotations
import shlex
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import LaunchConfig
from .ec2 import RegionContext, RegionLaunch, launch_with_retry


CANARY_RESOURCE = Path(__file__).parent / "resources" / "cluster_canary.py"
_REMOTE_CANARY = "/tmp/cluster_canary.py"  # noqa: S108 (well-known remote staging path)
_REMOTE_DATA_PREP = "/tmp/data_prep.sh"  # noqa: S108


def placement_group_name(cluster_id: str) -> str:
    """Deterministic cluster placement-group name for a cluster id."""
    return f"aws-bootstrap-cluster-{cluster_id}"


def node_alias(cluster_id: str, rank: int) -> str:
    """SSH-config alias for a cluster node (e.g. ``aws-ml1-0``)."""
    return f"aws-{cluster_id}-{rank}"


def nodes_to_add(current: int, target: int) -> int:
    """How many nodes to launch to reach ``target`` (never negative)."""
    return max(0, target - current)


def master_addr(nodes: list[dict]) -> str:
    """Private IP of rank 0 (the rendezvous/master node)."""
    rank0 = min(nodes, key=lambda n: n["Rank"])
    return rank0["PrivateIp"]


def build_torchrun_command(
    script: str,
    num_nodes: int,
    nproc_per_node: int,
    master_addr: str,
    rdzv_id: str,
    rdzv_port: int,
    script_args: list[str] | None = None,
) -> str:
    """The (identical-on-every-node) torchrun command using c10d rendezvous."""
    args = " ".join(script_args) if script_args else ""
    cmd = (
        f"torchrun --nnodes={num_nodes} --nproc-per-node={nproc_per_node} "
        f"--rdzv-backend=c10d --rdzv-endpoint={master_addr}:{rdzv_port} --rdzv-id={rdzv_id} "
        f"{script}"
    )
    return f"{cmd} {args}".rstrip()


def node_env(
    cluster_id: str,
    node_rank: int,
    num_nodes: int,
    num_gpus_per_node: int,
    node_ips: list[str],
    master_addr: str,
) -> dict[str, str]:
    """The AWSB_* environment contract injected on each node (Phase 3 escape hatch)."""
    return {
        "AWSB_CLUSTER_ID": cluster_id,
        "AWSB_NODE_RANK": str(node_rank),
        "AWSB_NUM_NODES": str(num_nodes),
        "AWSB_NUM_GPUS_PER_NODE": str(num_gpus_per_node),
        "AWSB_NODE_IPS": "\n".join(node_ips),
        "AWSB_MASTER_ADDR": master_addr,
    }


def render_node_config(env: dict[str, str]) -> str:
    """Serialize the ``AWSB_*`` env contract as a **sourceable** shell file.

    Each value is ``shlex.quote``d, so multi-line values (e.g. newline-joined
    ``AWSB_NODE_IPS``) stay inside a single quoted assignment rather than leaking
    a bare line that the shell would execute when the file is ``source``d.
    """
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())


def detect_version_skew(versions: dict[str, str]) -> list[str]:
    """Given ``{instance_id: version}``, return a list of mismatch descriptions.

    Empty list means all versions agree (or only one node). Nodes reporting an
    empty/None version are reported as mismatches (can't be verified).
    """
    present = {k: v for k, v in versions.items() if v}
    missing = [k for k, v in versions.items() if not v]
    distinct = set(present.values())
    mismatches: list[str] = []
    if len(distinct) > 1:
        mismatches.append("version mismatch across nodes: " + ", ".join(f"{k}={v}" for k, v in sorted(present.items())))
    mismatches.extend(f"{k}: version could not be determined" for k in sorted(missing))
    return mismatches


@dataclass
class ClusterNode:
    """A launched cluster node and its assigned rank."""

    rank: int
    launch: RegionLaunch


def launch_cluster_nodes(
    config: LaunchConfig,
    prepare_region: Callable[[str], RegionContext],
    count: int,
    start_rank: int,
    *,
    launch_fn: Callable[..., RegionLaunch] = launch_with_retry,
    on_node: Callable[[int, RegionLaunch], None] | None = None,
) -> list[ClusterNode]:
    """Launch ``count`` cluster nodes with ranks ``start_rank..start_rank+count-1``.

    Each node is launched via ``launch_fn`` (default
    :func:`ec2.launch_with_retry`). The caller's ``prepare_region`` pins the AZ
    and placement group on the returned :class:`RegionContext`. ``on_node`` fires
    after each successful node launch (e.g. to tag rank and add an SSH alias).
    """
    nodes: list[ClusterNode] = []
    for offset in range(count):
        rank = start_rank + offset
        launch = launch_fn(config, prepare_region)
        if on_node is not None:
            on_node(rank, launch)
        nodes.append(ClusterNode(rank=rank, launch=launch))
    return nodes


@dataclass
class NodeResult:
    """Result of running a command on one node."""

    instance_id: str
    rank: int
    returncode: int
    stdout: str
    stderr: str


def run_on_all_nodes(
    nodes: list[dict],
    command_for: Callable[[dict], str],
    *,
    run_fn: Callable[[dict, str], tuple[int, str, str]],
    max_workers: int | None = None,
) -> list[NodeResult]:
    """Run ``command_for(node)`` on every node concurrently; results in node order.

    Concurrency matters: torchrun nodes must start together to rendezvous.
    ``run_fn(node, command) -> (returncode, stdout, stderr)`` is injected (in
    production it wraps :func:`ssh.run_on_host`).
    """
    workers = max_workers or max(1, len(nodes))
    results: list[NodeResult | None] = [None] * len(nodes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_fn, node, command_for(node)): i for i, node in enumerate(nodes)}
        for future, i in futures.items():
            node = nodes[i]
            try:
                rc, out, err = future.result()
            except Exception as e:  # noqa: BLE001 — one node's crash must not abort the sweep un-attributed
                rc, out, err = 1, "", f"run failed on {node['InstanceId']}: {e}"
            results[i] = NodeResult(node["InstanceId"], node["Rank"], rc, out, err)
    return [r for r in results if r is not None]


def _job_command_for(
    node: dict,
    *,
    cluster_id: str,
    nodes: list[dict],
    nproc_per_node: int,
    rdzv_port: int,
    remote_script: str,
    script_args: list[str] | None,
) -> str:
    """Remote command for one node: activate the venv, then ``torchrun`` (``.py``)
    or an ``AWSB_*``-env-injected ``bash`` (``.sh`` escape hatch).

    ``~/.bashrc`` is not sourced for non-interactive ssh, so the venv must be
    activated explicitly to put ``torchrun``/``torch`` on PATH.
    """
    activate = "source ~/venv/bin/activate 2>/dev/null"
    addr = master_addr(nodes)
    if remote_script.endswith(".sh"):
        env = node_env(cluster_id, node["Rank"], len(nodes), nproc_per_node, [n["PrivateIp"] for n in nodes], addr)
        exports = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items())
        args = " ".join(script_args) if script_args else ""
        body = f"{exports} bash {remote_script} {args}".rstrip()
    else:
        body = build_torchrun_command(
            remote_script, len(nodes), nproc_per_node, addr, cluster_id, rdzv_port, script_args
        )
    return f"{activate}; {body}"


def run_distributed_job(
    nodes: list[dict],
    *,
    cluster_id: str,
    nproc_per_node: int,
    rdzv_port: int,
    local_script: Path,
    remote_script: str,
    script_args: list[str] | None,
    scp_fn: Callable[[dict, Path, str], bool],
    run_fn: Callable[[dict, str], tuple[int, str, str]],
    data_script: Path | None = None,
) -> list[NodeResult]:
    """Distribute ``local_script`` to every node and run it across the cluster.

    Steps: (1) SCP the script to all nodes; (2) if ``data_script`` is given, SCP
    and run it on every node in parallel (a barrier — training does not start
    until all preps succeed); (3) run the training command on all nodes in
    parallel (``torchrun`` for ``.py``; an ``AWSB_*``-env-injected ``bash`` for
    ``.sh``). Any SCP/prep failure short-circuits with failed ``NodeResult``s.
    ``scp_fn``/``run_fn`` are injected (production wraps the ``ssh`` primitives).
    """
    if any(not scp_fn(n, local_script, remote_script) for n in nodes):
        return [
            NodeResult(n["InstanceId"], n["Rank"], 1, "", f"failed to copy script to {n['InstanceId']}") for n in nodes
        ]

    if data_script is not None:
        if any(not scp_fn(n, data_script, _REMOTE_DATA_PREP) for n in nodes):
            return [
                NodeResult(n["InstanceId"], n["Rank"], 1, "", f"failed to copy data-prep to {n['InstanceId']}")
                for n in nodes
            ]
        prep_cmd = f"source ~/venv/bin/activate 2>/dev/null; bash {_REMOTE_DATA_PREP}"
        prep = run_on_all_nodes(nodes, lambda node: prep_cmd, run_fn=run_fn)
        if any(r.returncode != 0 for r in prep):
            return prep

    def command_for(node: dict) -> str:
        return _job_command_for(
            node,
            cluster_id=cluster_id,
            nodes=nodes,
            nproc_per_node=nproc_per_node,
            rdzv_port=rdzv_port,
            remote_script=remote_script,
            script_args=script_args,
        )

    return run_on_all_nodes(nodes, command_for, run_fn=run_fn)


def run_canary(
    nodes: list[dict],
    *,
    cluster_id: str,
    nproc_per_node: int,
    rdzv_port: int,
    scp_fn: Callable[[dict, Path, str], bool],
    run_fn: Callable[[dict, str], tuple[int, str, str]],
    canary_path: Path = CANARY_RESOURCE,
) -> list[NodeResult]:
    """Run the built-in canary — a distributed job with the shipped canary script."""
    return run_distributed_job(
        nodes,
        cluster_id=cluster_id,
        nproc_per_node=nproc_per_node,
        rdzv_port=rdzv_port,
        local_script=canary_path,
        remote_script=_REMOTE_CANARY,
        script_args=None,
        scp_fn=scp_fn,
        run_fn=run_fn,
    )
