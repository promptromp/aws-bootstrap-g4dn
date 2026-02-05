---
name: aws-bootstrap
description: >
  Use when the user wants to provision AWS GPU instances, check GPU instance status,
  terminate instances, manage EBS data volumes, or clean up cloud resources. Wraps the
  aws-bootstrap CLI for EC2 GPU instance lifecycle management (launch, status, terminate,
  cleanup, list).
---

# aws-bootstrap -- AWS GPU Instance Management

You have access to the `aws-bootstrap` CLI tool for provisioning and managing AWS EC2 GPU instances. Use it via the Bash tool. **Always use `--output json` when you need to parse results programmatically.**

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

EBS volumes are mounted at `/data`, survive spot interruptions, and persist independently of instances.

## Error Handling

- **Spot capacity errors**: The CLI auto-falls back to on-demand pricing
- **Quota limits** (`MaxSpotInstanceCountExceeded`, `VcpuLimitExceeded`): User needs to increase vCPU quota via AWS Service Quotas console
- **SSH timeouts**: Instance may still be initializing -- check `aws-bootstrap status`
- **No public IP**: Check VPC settings or assign an Elastic IP
- **EBS mount failures**: Non-fatal -- instance remains usable, may need manual mount

## Detailed Command Reference

See [commands.md](references/commands.md) for full option documentation and JSON output schemas.
