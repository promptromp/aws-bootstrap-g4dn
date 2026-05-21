# Multi-node distributed training on a cluster

This guide takes you from zero to a running multi-node PyTorch training job on a
cluster of GPU instances, using the `aws-bootstrap cluster` commands. It should
take ~15 minutes end to end (most of it waiting for spot capacity + remote
setup). Runnable example scripts live in [`examples/cluster/`](../examples/cluster/).

A **cluster** is just a set of GPU instances sharing a `--cluster-id` (an EC2
tag), launched into **one availability zone inside a cluster placement group**
so they can run a distributed `torchrun` job over fast intra-VPC networking.
AWS tags are the source of truth — there's no local state file.

The workflow is five commands:

```
launch  →  prepare  →  test  →  run  →  terminate
```

## Prerequisites

- The CLI installed and an AWS profile configured (see the main [README](../README.md)).
- **GPU vCPU quota** for the instance family you'll use. A 2-node `g5.xlarge`
  cluster needs **8 vCPUs** of the *G and VT* family. Check it, and request more
  if needed (per region):

  ```bash
  aws-bootstrap quota show    --family gvt --region us-west-2
  aws-bootstrap quota request --family gvt --type spot --desired-value 8 --region us-west-2
  ```

  New accounts often have **0** on-demand GPU quota and a small spot quota —
  request increases before your first run. See [EC2 vCPU Quotas](../README.md#ec2-vcpu-quotas).

## 1. Launch the cluster

```bash
aws-bootstrap cluster launch --cluster-id demo --nodes 2 \
    --instance-type g5.xlarge --region us-west-2 --wait
```

- All nodes land in **one AZ + a cluster placement group**; a self-referencing
  security-group rule lets them reach each other on the NCCL/rendezvous ports.
- Each node runs the **remote setup** (CUDA-matched PyTorch + `torchrun` into
  `~/venv`) — this is the slow part (a few minutes per node).
- Each node gets an SSH alias `aws-demo-0`, `aws-demo-1`, … (so `ssh aws-demo-0`
  just works), and a stable rank tag. Rank 0 is the rendezvous/master node.
- **Spot capacity:** `--wait` retries each node with backoff until capacity is
  available (instead of prompting for on-demand). If a type is capacity-out, try
  another (`g5.xlarge` vs `g4dn.12xlarge`, etc.) or another region with quota.
- **Grow it later:** re-run with a higher `--nodes` to add nodes incrementally.

Watch it:

```bash
aws-bootstrap cluster status --cluster-id demo --region us-west-2
```

## 2. Prepare (verify + canary)

```bash
aws-bootstrap cluster prepare --cluster-id demo --region us-west-2
```

`prepare` checks every node is reachable, has a GPU, and runs a **consistent
CUDA version** (it fails fast on skew), writes a per-node cluster config, then
runs a built-in **canary** — a tiny DDP all-reduce + a few SGD steps across all
nodes via `torchrun` c10d rendezvous. A passing canary means NCCL works across
your nodes and you're ready to train. Re-run the canary any time:

```bash
aws-bootstrap cluster test --cluster-id demo --region us-west-2
```

## 3. Run your training script

```bash
# synthetic data, no dataset needed — args after `--` go to your script:
aws-bootstrap cluster run --cluster-id demo --region us-west-2 \
    examples/cluster/train_ddp.py -- --epochs 3 --steps 50
```

`cluster run` copies your `SCRIPT` to every node and launches it across the
cluster:

- **`.py` script** → launched with `torchrun` (c10d rendezvous; `torchrun` sets
  `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR` for your code). This is the
  common case — write a normal DDP script (see `examples/cluster/train_ddp.py`).
- **`.sh` script** → run as-is with the **`AWSB_*` env contract** exported, so
  you can drive any launcher (`accelerate`, a custom `torchrun` line, Ray):
  `AWSB_NODE_RANK`, `AWSB_NUM_NODES`, `AWSB_NUM_GPUS_PER_NODE`, `AWSB_NODE_IPS`
  (newline-separated), `AWSB_MASTER_ADDR`.

Per-node stdout/stderr are saved to `.aws-bootstrap/clusters/demo/rank<N>.log`.
**Tip:** under c10d rendezvous, torch's global rank 0 (the rank your script
usually prints from) is *not* necessarily node rank 0 — the CLI echoes output
from every node, and you'll find the training output in whichever node's log got
global rank 0.

`--nproc-per-node N` controls processes (GPUs) per node (default 1; raise it for
multi-GPU instance types).

## 4. Stage data (optional `--data-script`)

For a real dataset, supply a data-prep script that runs **once per node, before
training** (a barrier — training won't start until every node finishes prep):

```bash
aws-bootstrap cluster run --cluster-id demo --region us-west-2 \
    --data-script examples/cluster/prepare_data.sh \
    examples/cluster/train_ddp.py -- --epochs 3 --data-dir /data/dataset
```

The recommended pattern (see `examples/cluster/prepare_data.sh`) is an
**idempotent** S3 pull into the per-node `/data` mount, guarded by a sentinel
file. Only the script's **exit code** is the success signal, so start it with
`set -euo pipefail`. `DistributedSampler` shards the per-node copy across ranks
at read time, so each node holding a full copy is correct at this scale. For
very large datasets, prefer streaming from S3 (S3 Connector for PyTorch /
Mountpoint) or a shared FSx for Lustre mount.

## 5. Tear it down

```bash
aws-bootstrap cluster terminate --cluster-id demo --region us-west-2 --yes
```

This terminates all nodes, removes their SSH aliases, waits for full
termination, then deletes the placement group. (Run `aws-bootstrap cluster
status` afterward to confirm nothing remains.)

## Caveats & notes

- **One AZ.** Cluster nodes are pinned to a single AZ for low-latency NCCL. A
  spot shortage in that AZ affects the whole cluster — use `--wait`, or start a
  fresh cluster id (which can land in a different AZ).
- **EFA is out of scope.** `g4dn`/`g5` instances run NCCL over ordinary VPC
  networking (EFA, for the lowest-latency collective comms, is essentially
  P-series only). This is fine for development, learning DDP, and modest-scale
  training; it is not tuned for large-scale LLM throughput.
- **Spot vs on-demand.** Spot is the default and cheapest. `cluster launch` does
  not silently fall back to on-demand — if spot is exhausted, re-run with
  `--wait` (retry) or `--on-demand` (and ensure you have on-demand quota).
- **Graduating beyond this.** For managed multi-cloud orchestration, gang
  scheduling, and large-scale fault tolerance, consider
  [SkyPilot](https://docs.skypilot.co/) or [Ray Train](https://docs.ray.io/en/latest/train/train.html);
  this tool deliberately stays a thin, transparent layer over EC2 you fully own.

## Command reference

Full options and JSON output schemas for every `cluster` subcommand are in the
agent skill reference:
[`aws-bootstrap-skill/skills/aws-bootstrap/references/commands.md`](../aws-bootstrap-skill/skills/aws-bootstrap/references/commands.md).
