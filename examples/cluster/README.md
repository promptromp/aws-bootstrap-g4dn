# Cluster training examples

Runnable examples for multi-node distributed training on an `aws-bootstrap`
cluster. Each file has a detailed usage header. Full walkthrough:
[`docs/multi-node-training.md`](../../docs/multi-node-training.md).

| File | What it is | Use with |
|------|------------|----------|
| [`train_ddp.py`](train_ddp.py) | A minimal multi-node PyTorch **DDP** training script (synthetic data by default, optional `--data-dir`). | `aws-bootstrap cluster run --cluster-id demo train_ddp.py -- --epochs 3` |
| [`prepare_data.sh`](prepare_data.sh) | An idempotent **per-node data-prep** script (S3 → `/data`), run before training. | `aws-bootstrap cluster run --cluster-id demo --data-script prepare_data.sh train_ddp.py -- --data-dir /data/dataset` |

End-to-end:

```bash
aws-bootstrap cluster launch  --cluster-id demo --nodes 2 --instance-type g5.xlarge --region us-west-2 --wait
aws-bootstrap cluster prepare --cluster-id demo --region us-west-2
aws-bootstrap cluster run     --cluster-id demo --region us-west-2 \
    examples/cluster/train_ddp.py -- --epochs 3 --steps 50
aws-bootstrap cluster terminate --cluster-id demo --region us-west-2 --yes
```
