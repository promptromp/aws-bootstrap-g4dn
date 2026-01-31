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
uv venv .venv
uv sync
direnv allow  # or manually: source .venv/bin/activate
```

This installs the `aws-bootstrap` CLI into your virtualenv.

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

## Additional Resources

For pricing information on GPU instances see [here](https://instances.vantage.sh/aws/ec2/g4dn.xlarge).
Deep Learning AMIs - see [here](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
Nvidia Nsight - Setup Remote Debugging - see [here](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Setup_Remote_Debugging.htm)

A couple of additional relevant recent tutorials (2025) for setting up CUDA environment on EC2 GPU instances are:

https://www.dolthub.com/blog/2025-03-12-provision-an-ec2-gpu-host-on-aws/
https://techfortalk.co.uk/2025/10/11/aws-ec2-setup-for-gpu-cuda-programming/
