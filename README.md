# aws-bootstrap-g4dn

This repository contains code and documentation to make it fast and easy to bootstrap an AWS EC2 instance running a Deep Learning AMI (e.g. Ubuntu or Amazon Linux) with a CUDA-compliant Nvidia GPU (e.g. g4dn.xlarge by default).

The idea is to make it easy in particular to quickly spawn cost-effective Spot Instances via AWS CLI , bootstrapping the instance with an SSH key, and ramping up to be able to develop using CUDA.

Main workflows we're optimizing for are hybrid local-remote workflows e.g.:

1. Using Jupyter server-client (with Jupyter server running on the instance and local jupyter client)
2. Using VSCode Remote SSH extension
3. Using Nvidia Nsight for remote debugging


## Requirements

1. AWS profile configured with relevant permissions (profile name can be passed via `--profile` or read from `AWS_PROFILE` env var)
2. AWS CLI v2 — see [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
3. Python 3.14+ and [uv](https://github.com/astral-sh/uv)
4. An SSH key pair (see below)

## Installation

```bash
git clone https://github.com/your-org/aws-bootstrap-g4dn.git
cd aws-bootstrap-g4dn
uv venv
uv sync
```

This installs the `aws-bootstrap` CLI into a virtualenv.

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

```bash
# Show available commands
aws-bootstrap --help

# Show launch options
aws-bootstrap launch --help

# Dry run — validates AMI lookup, key import, and security group without launching
aws-bootstrap launch --dry-run

# Launch a spot g4dn.xlarge (default)
aws-bootstrap launch

# Launch on-demand in a specific region with a custom instance type
aws-bootstrap launch --on-demand --instance-type g5.xlarge --region us-east-1

# Launch without running the remote setup script
aws-bootstrap launch --no-setup

# Use a specific AWS profile
aws-bootstrap launch --profile my-aws-profile
```

After launch, the CLI prints SSH and Jupyter tunnel commands:

```
ssh -i ~/.ssh/id_ed25519 ubuntu@<public-ip>
ssh -i ~/.ssh/id_ed25519 -NL 8888:localhost:8888 ubuntu@<public-ip>
```

### Managing Instances

```bash
# List all running aws-bootstrap instances
aws-bootstrap status

# List instances in a specific region
aws-bootstrap status --region us-east-1

# Terminate all aws-bootstrap instances (with confirmation prompt)
aws-bootstrap terminate

# Terminate specific instances
aws-bootstrap terminate i-abc123 i-def456

# Skip confirmation prompt
aws-bootstrap terminate --yes
```

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

For pricing information on GPU instances see [here](https://instances.vantage.sh/aws/ec2/g4dn.xlarge).
Spot Instances Quotas see [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html)
Deep Learning AMIs - see [here](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
Nvidia Nsight - Setup Remote Debugging - see [here](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Setup_Remote_Debugging.htm)

A couple of additional relevant recent tutorials (2025) for setting up CUDA environment on EC2 GPU instances are:

https://www.dolthub.com/blog/2025-03-12-provision-an-ec2-gpu-host-on-aws/
https://techfortalk.co.uk/2025/10/11/aws-ec2-setup-for-gpu-cuda-programming/
