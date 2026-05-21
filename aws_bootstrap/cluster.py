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


def placement_group_name(cluster_id: str) -> str:
    """Deterministic cluster placement-group name for a cluster id."""
    return f"aws-bootstrap-cluster-{cluster_id}"


def node_alias(cluster_id: str, rank: int) -> str:
    """SSH-config alias for a cluster node (e.g. ``aws-ml1-0``)."""
    return f"aws-{cluster_id}-{rank}"


def nodes_to_add(current: int, target: int) -> int:
    """How many nodes to launch to reach ``target`` (never negative)."""
    return max(0, target - current)
