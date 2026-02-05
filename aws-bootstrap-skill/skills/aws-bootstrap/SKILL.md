---
name: aws-bootstrap
description: >
  Use when the user wants to provision AWS GPU instances, check GPU instance status,
  terminate instances, manage EBS data volumes, or clean up cloud resources. Wraps the
  aws-bootstrap CLI for EC2 GPU instance lifecycle management (launch, status, terminate,
  cleanup, list).
---

# aws-bootstrap -- AWS GPU Instance Management

You have access to the `aws-bootstrap` CLI tool for provisioning and managing AWS EC2 GPU instances. Use it via the Bash tool. **Always pass `-o json` before the subcommand** (e.g., `aws-bootstrap -o json status`) **when you need to parse results programmatically.** The `--output`/`-o` flag is a global option and must come before the command name — placing it after (e.g., `aws-bootstrap status -o json`) will fail.

## Prerequisites

Before running any commands, verify:
1. The `aws-bootstrap` CLI is installed (`pip install aws-bootstrap-g4dn` or `uv pip install aws-bootstrap-g4dn`)
2. AWS credentials are configured (`AWS_PROFILE` env var or `--profile` flag)
3. An SSH key pair exists at `~/.ssh/id_ed25519` (or specify via `--key-path`)

You can check if the CLI is installed by running: `aws-bootstrap --version`

## Quick Reference

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `aws-bootstrap launch` | Provision a GPU instance (spot by default) | `--instance-type`, `--spot/--on-demand`, `--ebs-storage`, `--dry-run` |
| `aws-bootstrap status` | List running instances with IPs, pricing | `--gpu` (CUDA info), `--no-instructions` |
| `aws-bootstrap terminate` | Terminate instances and clean up | `[ID_OR_ALIAS...]`, `--keep-ebs`, `--yes` |
| `aws-bootstrap cleanup` | Remove stale SSH config + orphan EBS | `--include-ebs`, `--dry-run` |
| `aws-bootstrap list instance-types` | Browse GPU instance types | `--prefix` (default: g4dn) |
| `aws-bootstrap list amis` | Browse Deep Learning AMIs | `--filter` |
| `aws-bootstrap quota show` | Show GPU vCPU quotas (all families) | `--family` |
| `aws-bootstrap quota request` | Request a quota increase | `--family`, `--type`, `--desired-value`, `--yes` |
| `aws-bootstrap quota history` | Show quota increase request history | `--family`, `--type`, `--status` |

**Global options** (before the command): `--output json|yaml|table|text`, `--profile`, `--region`

## Structured Output

Always use `--output json` (aliased as `-o json`) when you need to process results:

```bash
# Get instance status as JSON
aws-bootstrap -o json status

# Dry-run launch to see what would happen
aws-bootstrap -o json launch --dry-run

# Terminate with --yes (required in structured output modes)
aws-bootstrap -o json terminate --yes
```

Commands requiring confirmation (`terminate`, `cleanup`) **must include `--yes`** when using `--output json/yaml/table`.

## Common Workflows

### Launch a GPU Instance

```bash
# Default: spot g4dn.xlarge in us-west-2
aws-bootstrap launch

# Specify instance type and region
aws-bootstrap launch --instance-type g5.xlarge --region us-east-1

# On-demand pricing (no spot interruption risk)
aws-bootstrap launch --on-demand

# With persistent EBS data volume (survives termination)
aws-bootstrap launch --ebs-storage 96

# Dry run first to validate configuration
aws-bootstrap launch --dry-run

# Custom Python version in remote venv
aws-bootstrap launch --python-version 3.13

# Non-default SSH port
aws-bootstrap launch --ssh-port 2222
```

After launch, the CLI:
1. Creates the instance (spot with auto-fallback to on-demand)
2. Adds an SSH alias (e.g. `aws-gpu1`) to `~/.ssh/config`
3. Runs remote setup (CUDA-matched PyTorch, Jupyter, GPU benchmark)
4. Mounts EBS volume at `/data` (if requested)

### Check Instance Status

```bash
# Human-readable status
aws-bootstrap status

# With GPU info (CUDA toolkit, driver version, GPU name)
aws-bootstrap status --gpu

# Machine-readable
aws-bootstrap -o json status
```

### Connect to an Instance

After launch, use the SSH alias printed in the output:

```bash
# Direct SSH (venv auto-activates)
ssh aws-gpu1

# Jupyter tunnel
ssh -NL 8888:localhost:8888 aws-gpu1
# Then open: http://localhost:8888

# VSCode Remote SSH
code --folder-uri vscode-remote://ssh-remote+aws-gpu1/home/ubuntu/workspace

# Run GPU benchmark
ssh aws-gpu1 'python ~/gpu_benchmark.py'
```

### Terminate and Clean Up

```bash
# Terminate by alias
aws-bootstrap terminate aws-gpu1

# Terminate all instances (with confirmation)
aws-bootstrap terminate

# Terminate but keep EBS volumes for reuse
aws-bootstrap terminate --keep-ebs

# Clean up stale SSH config entries
aws-bootstrap cleanup

# Also clean up orphan EBS volumes
aws-bootstrap cleanup --include-ebs

# Preview what would be cleaned (no changes)
aws-bootstrap cleanup --include-ebs --dry-run
```

### Persistent Data with EBS

```bash
# Create a new volume on launch
aws-bootstrap launch --ebs-storage 96

# After terminating with --keep-ebs, reattach to a new instance
aws-bootstrap terminate --keep-ebs
# Note the volume ID from output, then:
aws-bootstrap launch --ebs-volume-id vol-0abc123def456
```

EBS volumes are mounted at `/data`, survive spot interruptions, and persist independently of instances. Use `/data` for large datasets, model checkpoints, and training outputs — it persists across instance lifecycles while the root volume does not. For example:

```bash
# Store training data on persistent volume
ssh aws-gpu1 'mkdir -p /data/datasets /data/checkpoints /data/outputs'

# Download a dataset to persistent storage
ssh aws-gpu1 'cd /data/datasets && wget https://example.com/dataset.tar.gz'
```

## Remote Instance Environment

After launch and remote setup, each instance comes pre-configured with:

### Python Virtual Environment (`~/venv`)

- Located at `~/venv`, **auto-activated on SSH login** (via `~/.bashrc`)
- **PyTorch** is pre-installed with the correct CUDA wheel matching the host's CUDA toolkit version (e.g. `cu124`, `cu128`) — `torch.cuda.is_available()` works out of the box
- **torchvision** is also pre-installed with matching CUDA support
- Additional numerical/ML libraries from `requirements.txt`: `numpy`, `tqdm`, and other common dependencies
- Use `--python-version` on launch to pin a specific Python version (e.g. `3.13`)
- To install additional packages: `ssh aws-gpu1 'pip install transformers datasets'`

### GPU and CUDA

- NVIDIA drivers and CUDA toolkit are pre-installed via the Deep Learning AMI
- `nvidia-smi` and `nvcc` are available on PATH
- A GPU benchmark is pre-installed at `~/gpu_benchmark.py` (CNN + Transformer workloads)
- A Jupyter notebook for interactive GPU verification is at `~/gpu_smoke_test.ipynb`

### Jupyter

- JupyterLab runs as a systemd service on port 8888
- Access via SSH tunnel: `ssh -NL 8888:localhost:8888 aws-gpu1`, then open `http://localhost:8888`

### EBS Data Volume (`/data`)

If launched with `--ebs-storage` or `--ebs-volume-id`, a persistent gp3 EBS volume is mounted at `/data`. Use this for:
- **Large datasets** — download and store training data here so it persists across spot interruptions
- **Model checkpoints** — save checkpoints to `/data/checkpoints` to avoid losing training progress
- **Training outputs** — write logs, metrics, and results to `/data/outputs`

The `/data` volume is **not lost on spot interruption** — when AWS reclaims the instance, the volume detaches automatically and can be reattached to a new instance with `--ebs-volume-id`.

## Error Handling

- **Spot capacity errors**: The CLI auto-falls back to on-demand pricing
- **Quota limits** (`MaxSpotInstanceCountExceeded`, `VcpuLimitExceeded`): Check with `aws-bootstrap quota show` and request increases with `aws-bootstrap quota request --family gvt --type spot --desired-value 4`. Other families: `--family p5`, `--family p` (P4/P3/P2), `--family dl`
- **SSH timeouts**: Instance may still be initializing -- check `aws-bootstrap status`
- **No public IP**: Check VPC settings or assign an Elastic IP
- **EBS mount failures**: Non-fatal -- instance remains usable, may need manual mount

## Detailed Command Reference

See [commands.md](references/commands.md) for full option documentation and JSON output schemas.
