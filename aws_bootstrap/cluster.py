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
from dataclasses import dataclass

from .config import LaunchConfig
from .ec2 import RegionContext, RegionLaunch, launch_with_retry


def placement_group_name(cluster_id: str) -> str:
    """Deterministic cluster placement-group name for a cluster id."""
    return f"aws-bootstrap-cluster-{cluster_id}"


def node_alias(cluster_id: str, rank: int) -> str:
    """SSH-config alias for a cluster node (e.g. ``aws-ml1-0``)."""
    return f"aws-{cluster_id}-{rank}"


def nodes_to_add(current: int, target: int) -> int:
    """How many nodes to launch to reach ``target`` (never negative)."""
    return max(0, target - current)


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
