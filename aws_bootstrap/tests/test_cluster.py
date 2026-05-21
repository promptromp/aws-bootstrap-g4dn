"""Tests for cluster composition helpers (aws_bootstrap.cluster)."""

from __future__ import annotations

import pytest

from aws_bootstrap import cluster


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
