# aws-bootstrap-g4dn

--------------------------------------------------------------------------------

[![CI](https://github.com/promptromp/aws-bootstrap-g4dn/actions/workflows/ci.yml/badge.svg)](https://github.com/promptromp/aws-bootstrap-g4dn/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/promptromp/aws-bootstrap-g4dn)](https://github.com/promptromp/aws-bootstrap-g4dn/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/aws-bootstrap-g4dn)](https://pypi.org/project/aws-bootstrap-g4dn/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aws-bootstrap-g4dn)](https://pypi.org/project/aws-bootstrap-g4dn/)

One command to go from zero to a **fully configured GPU dev box** on AWS — with CUDA-matched PyTorch, Jupyter, SSH aliases, and a GPU benchmark ready to run.

```bash
aws-bootstrap launch          # Spot g4dn.xlarge in ~3 minutes
ssh aws-gpu1                  # You're in, venv activated, PyTorch works
```

### ✨ Key Features

| | Feature | Details |
|---|---|---|
| 🚀 | **One-command launch** | Spot (default) or on-demand, with automatic fallback on capacity errors |
| 🔑 | **Auto SSH config** | Adds `aws-gpu1` alias to `~/.ssh/config` — no IP juggling. Cleaned up on terminate |
| 🐍 | **CUDA-aware PyTorch** | Detects the installed CUDA toolkit (`nvcc`) and installs PyTorch from the matching wheel index — no more `torch.version.cuda` mismatches |
| ✅ | **PyTorch smoke test** | Runs a quick `torch.cuda` matmul after setup to verify the GPU stack works end-to-end |
| 📊 | **GPU benchmark included** | CNN (MNIST) + Transformer benchmarks with FP16/FP32/BF16 precision and tqdm progress |
| 📓 | **Jupyter ready** | Lab server auto-starts as a systemd service on port 8888 — just SSH tunnel and open |
| 🖥️ | **`status --gpu`** | Shows CUDA toolkit version, driver max, GPU architecture, spot pricing, uptime, and estimated cost |
| 🗑️ | **Clean terminate** | Stops instances, removes SSH aliases, shows shutting-down state until fully gone |

### 🎯 Target Workflows

1. **Jupyter server-client** — Jupyter runs on the instance, connect from your local browser
2. **VSCode Remote SSH** — `ssh aws-gpu1` just works with the Remote SSH extension
3. **NVIDIA Nsight remote debugging** — GPU debugging over SSH

---

## Requirements

1. AWS profile configured with relevant permissions (profile name can be passed via `--profile` or read from `AWS_PROFILE` env var)
2. AWS CLI v2 — see [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
3. Python 3.12+ and [uv](https://github.com/astral-sh/uv)
4. An SSH key pair (see below)

## Installation

### From PyPI

```bash
pip install aws-bootstrap-g4dn
```

### With uvx (no install needed)

[uvx](https://docs.astral.sh/uv/guides/tools/) runs the CLI directly in a temporary environment — no global install required:

```bash
uvx --from aws-bootstrap-g4dn aws-bootstrap launch
uvx --from aws-bootstrap-g4dn aws-bootstrap status
uvx --from aws-bootstrap-g4dn aws-bootstrap terminate
```

### From source (development)

```bash
git clone https://github.com/promptromp/aws-bootstrap-g4dn.git
cd aws-bootstrap-g4dn
uv venv
uv sync
```

All methods install the `aws-bootstrap` CLI.

## SSH Key Setup

The CLI expects an Ed25519 SSH public key at `~/.ssh/id_ed25519.pub` by default. If you don't have one, generate it:

```bash
ssh-keygen -t ed25519
```

Accept the default path (`~/.ssh/id_ed25519`) and optionally set a passphrase. The key pair will be imported into AWS automatically on first launch.

To use a different key, pass `--key-path`:

```bash
aws-bootstrap launch --key-path ~/.ssh/my_other_key.pub
```

## Usage

### 🚀 Launching an Instance

```bash
# Show available commands
aws-bootstrap --help

# Dry run — validates AMI lookup, key import, and security group without launching
aws-bootstrap launch --dry-run

# Launch a spot g4dn.xlarge (default)
aws-bootstrap launch

# Launch on-demand in a specific region with a custom instance type
aws-bootstrap launch --on-demand --instance-type g5.xlarge --region us-east-1

# Launch without running the remote setup script
aws-bootstrap launch --no-setup

# Use a specific Python version in the remote venv
aws-bootstrap launch --python-version 3.13

# Use a non-default SSH port
aws-bootstrap launch --ssh-port 2222

# Use a specific AWS profile
aws-bootstrap launch --profile my-aws-profile
```

After launch, the CLI:

1. **Adds an SSH alias** (e.g. `aws-gpu1`) to `~/.ssh/config`
2. **Runs remote setup** — installs utilities, creates a Python venv, installs CUDA-matched PyTorch, sets up Jupyter
3. **Runs a CUDA smoke test** — verifies `torch.cuda.is_available()` and runs a quick GPU matmul
4. **Prints connection commands** — SSH, Jupyter tunnel, GPU benchmark, and terminate

```bash
ssh aws-gpu1                  # venv auto-activates on login
```

### 🔧 What Remote Setup Does

The setup script runs automatically on the instance after SSH becomes available:

| Step | What |
|------|------|
| **GPU verify** | Confirms `nvidia-smi` and `nvcc` are working |
| **Utilities** | Installs `htop`, `tmux`, `tree`, `jq` |
| **Python venv** | Creates `~/venv` with `uv`, auto-activates in `~/.bashrc`. Use `--python-version` to pin a specific Python (e.g. `3.13`) |
| **CUDA-aware PyTorch** | Detects CUDA toolkit version → installs PyTorch from the matching `cu{TAG}` wheel index |
| **CUDA smoke test** | Runs `torch.cuda.is_available()` + GPU matmul to verify the stack |
| **GPU benchmark** | Copies `gpu_benchmark.py` to `~/gpu_benchmark.py` |
| **GPU smoke test notebook** | Copies `gpu_smoke_test.ipynb` to `~/gpu_smoke_test.ipynb` (open in JupyterLab) |
| **Jupyter** | Configures and starts JupyterLab as a systemd service on port 8888 |
| **SSH keepalive** | Configures server-side keepalive to prevent idle disconnects |

### 📊 GPU Benchmark

A GPU throughput benchmark is pre-installed at `~/gpu_benchmark.py` on every instance:

```bash
# Run both CNN and Transformer benchmarks (default)
ssh aws-gpu1 'python ~/gpu_benchmark.py'

# CNN only, quick run
ssh aws-gpu1 'python ~/gpu_benchmark.py --mode cnn --benchmark-batches 20'

# Transformer only with custom batch size
ssh aws-gpu1 'python ~/gpu_benchmark.py --mode transformer --transformer-batch-size 16'

# Run CUDA diagnostics first (tests FP16/FP32 matmul, autocast, etc.)
ssh aws-gpu1 'python ~/gpu_benchmark.py --diagnose'

# Force FP32 precision (if FP16 has issues on your GPU)
ssh aws-gpu1 'python ~/gpu_benchmark.py --precision fp32'
```

Reports: iterations/sec, samples/sec, peak GPU memory, and avg batch time for each model.

### 📓 Jupyter (via SSH Tunnel)

```bash
ssh -NL 8888:localhost:8888 aws-gpu1
# Then open: http://localhost:8888
```

Or with explicit key/IP:
```bash
ssh -i ~/.ssh/id_ed25519 -NL 8888:localhost:8888 ubuntu@<public-ip>
```

A **GPU smoke test notebook** (`~/gpu_smoke_test.ipynb`) is pre-installed on every instance. Open it in JupyterLab to interactively verify the CUDA stack, run FP32/FP16 matmuls, train a small CNN on MNIST, and visualise training loss and GPU memory usage.

### 📋 Listing Resources

```bash
# List all g4dn instance types (default)
aws-bootstrap list instance-types

# List a different instance family
aws-bootstrap list instance-types --prefix p3

# List Deep Learning AMIs (default filter)
aws-bootstrap list amis

# List AMIs with a custom filter
aws-bootstrap list amis --filter "ubuntu/images/hvm-ssd-gp3/ubuntu-noble*"

# Use a specific region
aws-bootstrap list instance-types --region us-east-1
aws-bootstrap list amis --region us-east-1
```

### 🖥️ Managing Instances

```bash
# Show all aws-bootstrap instances (including shutting-down)
aws-bootstrap status

# Include GPU info (CUDA toolkit + driver version, GPU name, architecture) via SSH
aws-bootstrap status --gpu

# Hide connection commands (shown by default for each running instance)
aws-bootstrap status --no-instructions

# List instances in a specific region
aws-bootstrap status --region us-east-1

# Terminate all aws-bootstrap instances (with confirmation prompt)
aws-bootstrap terminate

# Terminate specific instances
aws-bootstrap terminate i-abc123 i-def456

# Skip confirmation prompt
aws-bootstrap terminate --yes
```

`status --gpu` reports both the **installed CUDA toolkit** version (from `nvcc`) and the **maximum CUDA version supported by the driver** (from `nvidia-smi`), so you can see at a glance whether they match:

```
CUDA: 12.8 (driver supports up to 13.0)
```

SSH aliases are managed automatically — they're created on `launch`, shown in `status`, and cleaned up on `terminate`. Aliases use sequential numbering (`aws-gpu1`, `aws-gpu2`, etc.) and never reuse numbers from previous instances.

## EC2 vCPU Quotas

AWS accounts have [service quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) that limit how many vCPUs you can run per instance family. New or lightly-used accounts often have a **default quota of 0 vCPUs** for GPU instance families (G and VT), which will cause errors on launch:

- **Spot**: `MaxSpotInstanceCountExceeded`
- **On-Demand**: `VcpuLimitExceeded`

Check your current quotas (g4dn.xlarge requires at least 4 vCPUs):

```bash
# Spot G/VT quota
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-3819A6DF \
  --region us-west-2

# On-Demand G/VT quota
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2BBE81 \
  --region us-west-2
```

Request increases:

```bash
# Spot — increase to 4 vCPUs
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-3819A6DF \
  --desired-value 4 \
  --region us-west-2

# On-Demand — increase to 4 vCPUs
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2BBE81 \
  --desired-value 4 \
  --region us-west-2
```

Quota codes may vary by region or account type. To list the actual codes in your region:

```bash
# List all G/VT-related quotas
aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region us-west-2 \
  --query "Quotas[?contains(QuotaName, 'G and VT')].[QuotaCode,QuotaName,Value]" \
  --output table
```

Common quota codes:
- `L-3819A6DF` — All G and VT **Spot** Instance Requests
- `L-DB2BBE81` — Running **On-Demand** G and VT instances

Small increases (4-8 vCPUs) are typically auto-approved within minutes. You can also request increases via the [Service Quotas console](https://console.aws.amazon.com/servicequotas/home). While waiting, you can test the full launch/poll/SSH flow with a non-GPU instance type:

```bash
aws-bootstrap launch --instance-type t3.medium --ami-filter "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
```

## Additional Resources

| Topic | Link |
|-------|------|
| GPU instance pricing | [instances.vantage.sh](https://instances.vantage.sh/aws/ec2/g4dn.xlarge) |
| Spot instance quotas | [AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html) |
| Deep Learning AMIs | [AWS docs](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) |
| Nvidia Nsight remote debugging | [Nvidia docs](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Setup_Remote_Debugging.htm) |

Tutorials on setting up a CUDA environment on EC2 GPU instances:

- [Provision an EC2 GPU Host on AWS](https://www.dolthub.com/blog/2025-03-12-provision-an-ec2-gpu-host-on-aws/) (DoltHub, 2025)
- [AWS EC2 Setup for GPU/CUDA Programming](https://techfortalk.co.uk/2025/10/11/aws-ec2-setup-for-gpu-cuda-programming/) (TechForTalk, 2025)
