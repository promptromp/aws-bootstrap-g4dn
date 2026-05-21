#!/usr/bin/env bash
#
# Example per-node data-prep script for an aws-bootstrap cluster.
#
# Pass it to `cluster run --data-script`: it is copied to EVERY node and run
# ONCE, in parallel, BEFORE training starts (a barrier — training won't begin
# until all nodes finish prep). Each node prepares its own copy of the data, so
# DistributedSampler can shard it across ranks at read time.
#
# Only this script's EXIT CODE is the success signal, so it MUST start with
# `set -euo pipefail` (below) — otherwise a failed `aws s3 sync` could exit 0
# and training would start against missing/partial data.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
# 1. Edit S3_URI below to point at your dataset (or export DATASET_S3_URI).
# 2. Run a training job with this prep step:
#
#      aws-bootstrap cluster run --cluster-id demo \
#          --data-script examples/cluster/prepare_data.sh \
#          examples/cluster/train_ddp.py -- --epochs 3 --data-dir /data/dataset
#
#    The cluster nodes need an IAM role / credentials with s3:GetObject on the
#    bucket (the Deep Learning AMI has the AWS CLI preinstalled).
#
# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# * Idempotent: the `.prepared` sentinel makes re-runs (e.g. after adding a
#   node and re-running the job) a fast no-op.
# * Writes into /data — attach a persistent volume at launch (a future
#   per-node --ebs-storage; today /data is the instance's local disk) so the
#   dataset survives across jobs on the same cluster.
# * For very large datasets, prefer streaming from S3 in your DataLoader (the
#   S3 Connector for PyTorch / Mountpoint) or a shared FSx for Lustre mount
#   over copying to every node.

set -euo pipefail

S3_URI="${DATASET_S3_URI:-s3://my-bucket/my-dataset}"
DEST="/data/dataset"
SENTINEL="/data/.prepared"

if [ -f "$SENTINEL" ]; then
    echo "[prep] data already prepared at $DEST (sentinel $SENTINEL); skipping."
    exit 0
fi

echo "[prep] syncing $S3_URI -> $DEST ..."
mkdir -p "$DEST"
aws s3 sync "$S3_URI" "$DEST" --no-progress
touch "$SENTINEL"
echo "[prep] done: $(find "$DEST" -type f | wc -l) file(s) under $DEST"
