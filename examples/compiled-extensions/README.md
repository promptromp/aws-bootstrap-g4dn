# Compiled extensions

Packages such as flash-attn, causal-conv1d and mamba-ssm build CUDA kernels
against the PyTorch that is already installed. On an `aws-bootstrap` instance that
means building inside `~/venv` with `--no-build-isolation`, which can take many
minutes per package.

| File | What it is |
|------|------------|
| [`install_compiled.sh`](install_compiled.sh) | Builds and installs such packages into `~/venv`, with an optional S3 wheel cache keyed by what the wheels were built against (torch, its CUDA, the nvcc toolkit, Python, CPU architecture, libc, GPU generation). Usage and settings are in its header. |

```bash
# One instance, no cache
ssh aws-gpu1 'bash -s -- flash-attn==2.8.3' < examples/compiled-extensions/install_compiled.sh

# Share built wheels between instances (the instance needs S3 access: launch with --iam-instance-profile)
ssh aws-gpu1 'WHEEL_CACHE=s3://my-bucket/wheels bash -s -- causal-conv1d==1.5.0 mamba-ssm==2.2.4' \
    < examples/compiled-extensions/install_compiled.sh

# Every node of a cluster: build once on the first node, then the rest install from the cache
cmd='WHEEL_CACHE=s3://my-bucket/wheels bash -s -- flash-attn==2.8.3'
ssh aws-ml1-0 "$cmd" < examples/compiled-extensions/install_compiled.sh
for host in aws-ml1-1 aws-ml1-2; do
    ssh "$host" "$cmd" < examples/compiled-extensions/install_compiled.sh &
done; wait
```

pip decides whether a cached wheel satisfies each spec, so specifiers work as usual; pin with
`==` for reproducibility (an unpinned name reuses the newest cached version). URLs and paths
always build and are never cached. List packages that need compiling in dependency order
(causal-conv1d before mamba-ssm).
