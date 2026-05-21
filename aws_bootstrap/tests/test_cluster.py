"""Tests for cluster composition helpers (aws_bootstrap.cluster)."""

from __future__ import annotations
from unittest.mock import MagicMock

import pytest

from aws_bootstrap import cluster
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
            placement_az="us-east-1c",
            placement_group="aws-bootstrap-cluster-ml1",
        )
        return RegionLaunch("us-east-1", ctx, {"InstanceId": f"i-{i}"}, "spot")

    return launch_fn


def test_launch_cluster_nodes_assigns_sequential_ranks():
    holder = {"n": 0}
    seen = []
    results = cluster.launch_cluster_nodes(
        LaunchConfig(regions=("us-east-1",)),
        prepare_region=lambda r: None,
        count=3,
        start_rank=0,
        launch_fn=_fake_launch(holder),
        on_node=lambda rank, launch: seen.append((rank, launch.instance["InstanceId"])),
    )
    assert [r.rank for r in results] == [0, 1, 2]
    assert [r.launch.instance["InstanceId"] for r in results] == ["i-0", "i-1", "i-2"]
    assert seen == [(0, "i-0"), (1, "i-1"), (2, "i-2")]


def test_launch_cluster_nodes_starts_at_offset():
    holder = {"n": 0}
    results = cluster.launch_cluster_nodes(
        LaunchConfig(regions=("us-east-1",)),
        prepare_region=lambda r: None,
        count=2,
        start_rank=2,
        launch_fn=_fake_launch(holder),
    )
    assert [r.rank for r in results] == [2, 3]
