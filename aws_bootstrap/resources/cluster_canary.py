"""Tiny distributed-training canary for aws-bootstrap clusters.

Run across all nodes via torchrun (c10d rendezvous). Verifies the process group
forms, NCCL/gloo all-reduce is correct, a few SGD steps run on every rank, and
prints per-rank GPU + the elected master. Exits non-zero on any failure.
"""

from __future__ import annotations
import os

import torch
import torch.distributed as dist
from torch import nn


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        gpu_name = torch.cuda.get_device_name(local_rank)
    else:
        device = torch.device("cpu")
        gpu_name = "cpu"

    # All-reduce correctness: sum of (rank+1) over all ranks == world*(world+1)/2.
    t = torch.full((1,), float(rank + 1), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size + 1) / 2
    ok = abs(t.item() - expected) < 1e-3

    # A few SGD steps so every rank exercises autograd + comms.
    model: nn.Module = nn.Linear(8, 8).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3):
        x = torch.randn(16, 8, device=device)
        y = model(x).sum()
        opt.zero_grad()
        y.backward()
        opt.step()

    print(
        f"[canary] rank={rank}/{world_size} local_rank={local_rank} "
        f"gpu={gpu_name} master={os.environ.get('MASTER_ADDR', '?')} all_reduce_ok={ok}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(f"all-reduce check failed on rank {rank}: got {t.item()}, expected {expected}")


if __name__ == "__main__":
    main()
