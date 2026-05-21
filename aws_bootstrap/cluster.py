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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import LaunchConfig
from .ec2 import RegionContext, RegionLaunch, launch_with_retry


CANARY_RESOURCE = Path(__file__).parent / "resources" / "cluster_canary.py"
_REMOTE_CANARY = "/tmp/cluster_canary.py"  # noqa: S108 (well-known remote staging path)


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
            rc, out, err = future.result()
            node = nodes[i]
            results[i] = NodeResult(node["InstanceId"], node["Rank"], rc, out, err)
    return [r for r in results if r is not None]


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
    """SCP the canary to every node, then run torchrun on all nodes in parallel.

    ``scp_fn(node, local, remote) -> bool`` and ``run_fn(node, command)`` are
    injected (production wraps :func:`ssh.scp_to_host` / :func:`ssh.run_on_host`).
    If SCP fails on any node, torchrun is not started and every node is reported
    as failed.
    """
    if any(not scp_fn(n, canary_path, _REMOTE_CANARY) for n in nodes):
        return [
            NodeResult(n["InstanceId"], n["Rank"], 1, "", f"failed to copy canary to {n['InstanceId']}") for n in nodes
        ]

    addr = master_addr(nodes)
    command = build_torchrun_command(
        script=_REMOTE_CANARY,
        num_nodes=len(nodes),
        nproc_per_node=nproc_per_node,
        master_addr=addr,
        rdzv_id=cluster_id,
        rdzv_port=rdzv_port,
    )
    # Activate the remote venv so torchrun/torch are on PATH (~/.bashrc is not
    # sourced for non-interactive ssh).
    wrapped = f"source ~/venv/bin/activate 2>/dev/null; {command}"
    return run_on_all_nodes(nodes, lambda node: wrapped, run_fn=run_fn)
