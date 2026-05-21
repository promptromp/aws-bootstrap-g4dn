"""Centralized infrastructure & tagging constants.

User-facing *defaults* (region, wait timeout, instance type, volume size,
key/SG/alias names, AMI filter) live in :mod:`aws_bootstrap.config` and are
re-exported here so there is a single import surface. This module additionally
holds the lower-level infra literals (tag keys, AWS resource types, ports,
device names, waiter configs, AMI owner account IDs) that were previously
duplicated as string/number literals across modules.
"""

from __future__ import annotations

from .config import (  # noqa: F401  (re-export: single import surface)
    DEFAULT_ALIAS_PREFIX,
    DEFAULT_REGION,
    DEFAULT_SSH_PORT,
    DEFAULT_TAG_VALUE,
    DEFAULT_WAIT_TIMEOUT,
)


# Alias: the `created-by` tag value identifying tool-managed resources.
TAG_VALUE = DEFAULT_TAG_VALUE
# SSH config alias prefix (aws-gpu1, aws-gpu2, …).
ALIAS_PREFIX = DEFAULT_ALIAS_PREFIX

# --- Tagging -----------------------------------------------------------------
# Resource-discovery depends on exact tag keys; a typo silently breaks
# status/terminate/cleanup, so these must come from one place.
TAG_CREATED_BY = "created-by"  # value = LaunchConfig.tag_value
TAG_NAME = "Name"
TAG_BOOTSTRAP_INSTANCE = "aws-bootstrap-instance"  # links a data volume to its instance
TAG_CLUSTER_ID = "aws-bootstrap-cluster"  # value = the user's --cluster-id
TAG_CLUSTER_RANK = "aws-bootstrap-cluster-rank"  # value = stable node index "0".."N-1"

# --- AWS resource types (TagSpecifications) ----------------------------------
RES_INSTANCE = "instance"
RES_VOLUME = "volume"
RES_SECURITY_GROUP = "security-group"
RES_KEY_PAIR = "key-pair"

# --- Networking --------------------------------------------------------------
SSH_PORT_DEFAULT = DEFAULT_SSH_PORT
JUPYTER_PORT = 8888
SSH_INGRESS_CIDR = "0.0.0.0/0"  # public SSH ingress (intentional for remote dev)
SSH_CONNECT_TIMEOUT = 10  # seconds, ssh -o ConnectTimeout / socket timeout
RDZV_PORT = 29400  # torchrun c10d rendezvous port (intra-cluster)

# --- Storage -----------------------------------------------------------------
VOLUME_TYPE = "gp3"
ROOT_DEVICE_NAME = "/dev/sda1"
EBS_DEVICE_NAME = "/dev/sdf"
EBS_MOUNT_POINT = "/data"

# --- EC2 waiters -------------------------------------------------------------
# Two distinct profiles, kept separate on purpose:
#  * volume attach/availability is fast (≈2 min budget)
#  * detach during `terminate` can be slow because the instance is shutting
#    down, so it gets a longer (≈5 min) budget.
INSTANCE_RUNNING_WAITER = {"Delay": 10, "MaxAttempts": 60}  # ≈10 min
INSTANCE_STATUS_OK_WAITER = {"Delay": 15, "MaxAttempts": 60}  # ≈15 min
EBS_VOLUME_WAITER = {"Delay": 5, "MaxAttempts": 24}  # ≈2 min (attach/available)
EBS_DETACH_WAITER = {"Delay": 10, "MaxAttempts": 30}  # ≈5 min (detach on terminate)

# --- AMI owner account IDs ---------------------------------------------------
# Public, well-known vendor account IDs (NOT this account's ID).
AMI_OWNER_AMAZON = "amazon"  # AWS Deep Learning / Amazon Linux AMIs
AMI_OWNER_CANONICAL = "099720109477"  # Canonical (official Ubuntu AMIs)
AMI_OWNER_RHEL = "309956199498"  # Red Hat (official RHEL AMIs)
