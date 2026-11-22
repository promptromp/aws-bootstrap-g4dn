# Compiled extensions

Packages such as flash-attn, causal-conv1d, mamba-ssm and apex build CUDA kernels
against the PyTorch that is already installed. On an `aws-bootstrap` instance that
means building inside `~/venv` with `--no-build-isolation`, which can take many
minutes per package.

| File | What it is |
|------|------------|
| [`install_compiled.sh`](install_compiled.sh) | Builds and installs such packages into `~/venv`, with an optional S3 wheel cache keyed by torch/CUDA, Python, CPU architecture and GPU generation. Usage and settings are in its header. |

```bash
# One instance, no cache
ssh aws-gpu1 'bash -s -- causal-conv1d==1.5.0' < examples/compiled-extensions/install_compiled.sh

# Share built wheels between instances (the instance needs S3 access: launch with --iam-instance-profile)
ssh aws-gpu1 'WHEEL_CACHE=s3://my-bucket/wheels bash -s -- causal-conv1d==1.5.0 mamba-ssm==2.2.4' \
    < examples/compiled-extensions/install_compiled.sh

# Every node of a cluster
for host in aws-ml1-0 aws-ml1-1; do
    ssh "$host" 'WHEEL_CACHE=s3://my-bucket/wheels bash -s -- causal-conv1d==1.5.0' \
        < examples/compiled-extensions/install_compiled.sh &
done; wait
```

Pin versions with `==`: a pinned package reuses only a cached wheel of that version.
List packages that need compiling in dependency order (causal-conv1d before mamba-ssm).
