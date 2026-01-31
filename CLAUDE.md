# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

aws-bootstrap-g4dn is a Python CLI tool (`aws-bootstrap`) that bootstraps AWS EC2 GPU instances (g4dn.xlarge default) running Deep Learning AMIs for hybrid local-remote development. It provisions cost-effective Spot Instances via boto3 with SSH key import, security group setup, instance polling, and remote setup automation.

Target workflows: Jupyter server-client, VSCode Remote SSH, and NVIDIA Nsight remote debugging.

## Tech Stack & Requirements

- **Python 3.14+** with **uv** package manager (astral-sh/uv) — used for venv creation, dependency management, and running the project
- **boto3** — AWS SDK for EC2 provisioning (AMI lookup, security groups, instance launch, waiters)
- **click** — CLI framework with built-in color support (`click.secho`, `click.style`)
- **hatchling** — build backend (configured in pyproject.toml)
- **AWS CLI v2** with a configured AWS profile (`AWS_PROFILE` env var or `--profile` flag)
- **direnv** for automatic venv activation (`.envrc` sources `.venv/bin/activate`)

## Development Setup

```bash
uv venv .venv
uv sync
direnv allow  # or manually: source .venv/bin/activate
```

## Project Structure

```
aws_bootstrap/
    __init__.py          # Package init
    cli.py               # Click CLI entry point (launch, status, terminate commands)
    config.py            # LaunchConfig dataclass with defaults
    ec2.py               # AMI lookup, security group, instance launch/find/terminate, polling, spot pricing
    ssh.py               # SSH key pair import, SSH readiness check, remote setup, ~/.ssh/config management
    resources/           # Non-Python artifacts SCP'd to remote instances
        __init__.py
        gpu_benchmark.py # GPU throughput benchmark (CNN + Transformer), copied to ~/gpu_benchmark.py on instance
        remote_setup.sh  # Uploaded & run on instance post-boot (GPU verify, Jupyter, etc.)
        requirements.txt # Python dependencies installed on the remote instance
    tests/               # Unit tests (pytest)
        test_config.py
        test_cli.py
        test_ec2.py
        test_ssh_config.py
        test_ssh_gpu.py
```

Entry point: `aws-bootstrap = "aws_bootstrap.cli:main"` (installed via `uv sync`)

## CLI Commands

- **`launch`** — provisions an EC2 instance (spot by default, falls back to on-demand on capacity errors); adds SSH config alias (e.g. `aws-gpu1`) to `~/.ssh/config`
- **`status`** — lists active instances with type, IP, SSH alias, pricing (spot price/hr or on-demand), uptime, and estimated cost for running spot instances; `--gpu` flag queries GPU info (CUDA version, driver, GPU name/architecture) via SSH
- **`terminate`** — terminates instances by ID or all aws-bootstrap instances in the region; removes SSH config aliases
- **`list instance-types`** — lists EC2 instance types matching a family prefix (default: `g4dn`), showing vCPUs, memory, and GPU info
- **`list amis`** — lists available AMIs matching a name pattern (default: Deep Learning Base OSS Nvidia Driver GPU AMIs), sorted newest-first

## Coding Conventions

- **Linting**: `ruff check` — line length 120, rules: E, F, UP, B, SIM, I, PLC
- **Formatting**: `ruff format` — double quotes, isort via ruff
- **Type checking**: `mypy` with `ignore_missing_imports = true`
- **Testing**: `pytest`
- **All-in-one**: `pre-commit run --all` runs the full chain (ruff check, ruff format, mypy, pytest)

After making changes, run:

```bash
pre-commit run --all
```

Or run tools individually:

```bash
uv run ruff check aws_bootstrap/
uv run ruff format aws_bootstrap/
uv run mypy aws_bootstrap/
uv run pytest
```

Use `uv add <package>` to add dependencies and `uv add --group dev <package>` for dev dependencies.

## CUDA-Aware PyTorch Installation

`remote_setup.sh` detects the CUDA toolkit version on the instance (via `nvcc`, falling back to `nvidia-smi`) and installs PyTorch from the matching CUDA wheel index (`https://download.pytorch.org/whl/cu{TAG}`). This ensures `torch.version.cuda` matches the system's CUDA toolkit, which is required for compiling custom CUDA extensions with `nvcc`.

The `KNOWN_CUDA_TAGS` array in `remote_setup.sh` lists the CUDA wheel tags published by PyTorch (e.g., `118 121 124 126 128 129 130`). When PyTorch adds support for a new CUDA version, add the corresponding tag to this array. Check available tags at: https://download.pytorch.org/whl/

`torch` and `torchvision` are **not** in `resources/requirements.txt` — they are installed separately by the CUDA detection logic in `remote_setup.sh`. All other Python dependencies remain in `requirements.txt`.

## GPU Benchmark

`resources/gpu_benchmark.py` is uploaded to `~/gpu_benchmark.py` on the remote instance during setup. It benchmarks GPU throughput with two modes: CNN on MNIST and a GPT-style Transformer on synthetic data. It reports samples/sec, batch times, and peak GPU memory. Supports `--precision` (fp32/fp16/bf16/tf32), `--diagnose` for CUDA smoke tests, and separate `--transformer-batch-size` (default 32, T4-safe). Dependencies (`torch`, `torchvision`, `tqdm`) are already installed by the setup script.

## Keeping Docs Updated

When making changes that affect project setup, CLI interface, dependencies, project structure, or development workflows, update **README.md** and **CLAUDE.md** accordingly:

- **README.md** — user-facing: installation, usage examples, CLI options, AWS setup/quota instructions
- **CLAUDE.md** — agent-facing: project overview, tech stack, project structure, coding conventions
