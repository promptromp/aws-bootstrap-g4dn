"""Example multi-node DistributedDataParallel (DDP) training script.

Run it across an aws-bootstrap cluster with ``cluster run``. The cluster
launches it on every node via ``torchrun`` (c10d rendezvous), which sets the
standard distributed env vars this script reads: ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR``. You do NOT set them yourself.

----------------------------------------------------------------------------
QUICK START (copy/paste)
----------------------------------------------------------------------------
# 0. (one-time) make sure you have GPU vCPU quota in your region. 2 nodes of
#    g5.xlarge = 8 vCPUs of the "G and VT" family. Check / request:
#      aws-bootstrap quota show --family gvt --region us-west-2
#      aws-bootstrap quota request --family gvt --type spot --desired-value 8 --region us-west-2

# 1. Launch a 2-node spot cluster (use --wait to ride out spot capacity):
aws-bootstrap cluster launch --cluster-id demo --nodes 2 \
    --instance-type g5.xlarge --region us-west-2 --wait

# 2. Verify it (GPU + consistent CUDA + a distributed canary):
aws-bootstrap cluster prepare --cluster-id demo

# 3. Run THIS script across the cluster on synthetic data (no dataset needed).
#    Everything after `--` is passed to the script:
aws-bootstrap cluster run --cluster-id demo examples/cluster/train_ddp.py -- --epochs 3 --steps 50

# 4. ...or train on a real dataset staged to /data by a data-prep step first
#    (see examples/cluster/prepare_data.sh):
aws-bootstrap cluster run --cluster-id demo \
    --data-script examples/cluster/prepare_data.sh \
    examples/cluster/train_ddp.py -- --epochs 3 --data-dir /data/dataset

# 5. Tear everything down when you're done:
aws-bootstrap cluster terminate --cluster-id demo --yes

----------------------------------------------------------------------------
NOTES
----------------------------------------------------------------------------
* Per-node stdout/stderr are saved locally under
  ``.aws-bootstrap/clusters/demo/rank<N>.log``. Under c10d rendezvous, torch's
  global rank 0 (which prints below) is not necessarily node rank 0, so look
  across the rank logs for the training output.
* Checkpoints are written to ``/data/checkpoints`` when a ``/data`` volume is
  mounted (launch with ``--ebs-storage`` / ``--ebs-volume-id`` on a future
  per-node EBS-enabled launch), else to the local disk.
* The model/dataset here are intentionally tiny — this is a plumbing example,
  not a model you'd actually ship.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class SyntheticDataset(Dataset):
    """Random (x, y) pairs so the example runs with no external data."""

    def __init__(self, n: int, dim: int) -> None:
        self.x = torch.randn(n, dim)
        self.y = torch.randint(0, 2, (n,))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def build_dataset(data_dir: str | None, dim: int) -> Dataset:
    """Load a dataset from ``data_dir`` if given, else synthetic data.

    Replace this with your real loader (e.g. ``torchvision.datasets.ImageFolder``
    or a tokenized corpus) reading from the ``/data`` volume your data-prep
    script populated.
    """
    if data_dir:
        path = Path(data_dir)
        if not path.exists():
            raise SystemExit(f"--data-dir {path} not found (did the --data-script run and mount /data?)")
        # Example placeholder: a real script would parse files under `path`.
        print(f"[train] loading dataset from {path} ...", flush=True)
    return SyntheticDataset(n=4096, dim=dim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=0, help="Cap steps/epoch (0 = full pass).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--data-dir", default=None, help="Dataset dir (e.g. /data/dataset); synthetic if omitted.")
    args = parser.parse_args()

    # torchrun provides these. Default to single-process if run directly.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dataset = build_dataset(args.data_dir, args.dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model: nn.Module = nn.Sequential(nn.Linear(args.dim, 256), nn.ReLU(), nn.Linear(256, 2)).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # so each epoch reshuffles deterministically across ranks
        for step, (x, y) in enumerate(loader):
            if args.steps and step >= args.steps:
                break
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if rank == 0 and step % 10 == 0:
                print(f"[train] epoch {epoch} step {step:3d} loss {loss.item():.4f}", flush=True)

    if rank == 0:
        ckpt_dir = Path("/data/checkpoints") if Path("/data").is_dir() else Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = ckpt_dir / "model.pt"
        torch.save(model.module.state_dict(), ckpt)
        gpu = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else "cpu"
        print(f"[train] DONE world_size={world_size} backend={backend} gpu={gpu} checkpoint={ckpt}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
