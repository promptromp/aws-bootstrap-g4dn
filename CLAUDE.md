# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

aws-bootstrap-g4dn is a Python CLI tool (`aws-bootstrap`) that bootstraps AWS EC2 GPU instances (g4dn.xlarge default) running Deep Learning AMIs for hybrid local-remote development. It provisions cost-effective Spot Instances via boto3 with SSH key import, security group setup, instance polling, and remote setup automation.

Target workflows: Jupyter server-client, VSCode Remote SSH, and NVIDIA Nsight remote debugging.

## Tech Stack & Requirements

- **Python 3.12+** with **uv** package manager (astral-sh/uv) — used for venv creation, dependency management, and running the project
- **boto3** — AWS SDK for EC2 provisioning (AMI lookup, security groups, instance launch, waiters)
- **click** — CLI framework with built-in color support (`click.secho`, `click.style`)
- **setuptools + setuptools-scm** — build backend with git-tag-based versioning (configured in pyproject.toml)
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
    gpu.py               # GPU architecture mapping and GpuInfo dataclass
    ssh.py               # SSH key pair import, SSH readiness check, remote setup, ~/.ssh/config management, GPU queries
    resources/           # Non-Python artifacts SCP'd to remote instances
        __init__.py
        gpu_benchmark.py       # GPU throughput benchmark (CNN + Transformer), copied to ~/gpu_benchmark.py on instance
        gpu_smoke_test.ipynb   # Interactive Jupyter notebook for GPU verification, copied to ~/gpu_smoke_test.ipynb
        launch.json            # VSCode CUDA debug config template (deployed to ~/workspace/.vscode/launch.json)
        saxpy.cu               # Example CUDA SAXPY source (deployed to ~/workspace/saxpy.cu)
        tasks.json             # VSCode CUDA build tasks template (deployed to ~/workspace/.vscode/tasks.json)
        remote_setup.sh        # Uploaded & run on instance post-boot (GPU verify, Jupyter, etc.)
        requirements.txt       # Python dependencies installed on the remote instance
    tests/               # Unit tests (pytest)
        test_config.py
        test_cli.py
        test_ec2.py
        test_gpu.py
        test_ssh_config.py
        test_ssh_gpu.py
docs/
    nsight-remote-profiling.md # Nsight Compute, Nsight Systems, and Nsight VSCE remote profiling guide
    spot-request-lifecycle.md  # Research notes on spot request cleanup
```

Entry point: `aws-bootstrap = "aws_bootstrap.cli:main"` (installed via `uv sync`)

## CLI Commands

- **`launch`** — provisions an EC2 instance (spot by default, falls back to on-demand on capacity errors); adds SSH config alias (e.g. `aws-gpu1`) to `~/.ssh/config`; `--python-version` controls which Python `uv` installs in the remote venv; `--ssh-port` overrides the default SSH port (22) for security group ingress, connection checks, and SSH config
- **`status`** — lists all non-terminated instances (including `shutting-down`) with type, IP, SSH alias, pricing (spot price/hr or on-demand), uptime, and estimated cost for running spot instances; `--gpu` flag queries GPU info via SSH, reporting both CUDA toolkit version (from `nvcc`) and driver-supported max (from `nvidia-smi`); `--instructions` (default: on) prints connection commands (SSH, Jupyter tunnel, VSCode Remote SSH, GPU benchmark) for each running instance; suppress with `--no-instructions`
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

## Remote Setup Details

`remote_setup.sh` also:
- Creates `~/venv` and appends `source ~/venv/bin/activate` to `~/.bashrc` so the venv is auto-activated on SSH login. When `--python-version` is passed to `launch`, the CLI sets `PYTHON_VERSION` as an inline env var on the SSH command; `remote_setup.sh` reads it to run `uv python install` and `uv venv --python` with the requested version
- Adds NVIDIA Nsight Systems (`nsys`) to PATH if installed under `/opt/nvidia/nsight-systems/` (pre-installed on Deep Learning AMIs but not on PATH by default). Fixes directory permissions, finds the latest version, and prepends its `bin/` to PATH in `~/.bashrc`
- Runs a quick CUDA smoke test (`torch.cuda.is_available()` + GPU matmul) after PyTorch installation to verify the GPU stack; prints a WARNING on failure but does not abort
- Copies `gpu_benchmark.py` to `~/gpu_benchmark.py` and `gpu_smoke_test.ipynb` to `~/gpu_smoke_test.ipynb`
- Sets up `~/workspace/.vscode/` with `launch.json` and `tasks.json` for CUDA debugging. Detects `cuda-gdb` path and GPU SM architecture (via `nvidia-smi --query-gpu=compute_cap`) at deploy time, replacing `__CUDA_GDB_PATH__` and `__GPU_ARCH__` placeholders in the template files via `sed`

## GPU Benchmark

`resources/gpu_benchmark.py` is uploaded to `~/gpu_benchmark.py` on the remote instance during setup. It benchmarks GPU throughput with two modes: CNN on MNIST and a GPT-style Transformer on synthetic data. It reports samples/sec, batch times, and peak GPU memory. Supports `--precision` (fp32/fp16/bf16/tf32), `--diagnose` for CUDA smoke tests, and separate `--transformer-batch-size` (default 32, T4-safe). Dependencies (`torch`, `torchvision`, `tqdm`) are already installed by the setup script.

## Versioning & Publishing

Version is derived automatically from git tags via **setuptools-scm** — no hardcoded version string in the codebase.

- **Tagged commits** (e.g. `0.1.0`) produce exact versions
- **Between tags**, setuptools-scm generates dev versions like `0.1.1.dev5+gabcdef` (valid PEP 440)
- `click.version_option(package_name="aws-bootstrap-g4dn")` in `cli.py` reads from package metadata — works automatically

### Release process

1. Create and push a git tag: `git tag X.Y.Z && git push origin X.Y.Z`
2. The `publish-to-pypi.yml` workflow triggers on tag push and:
   - Builds wheel + sdist
   - Publishes to PyPI and TestPyPI via OIDC trusted publishing
   - Creates a GitHub Release with Sigstore-signed artifacts

### Required one-time setup (repo owner)

- **PyPI trusted publisher**: https://pypi.org/manage/account/publishing/ — add publisher for `aws-bootstrap-g4dn`, workflow `publish-to-pypi.yml`, environment `pypi`
- **TestPyPI trusted publisher**: same at https://test.pypi.org/manage/account/publishing/, environment `testpypi`
- **GitHub environments**: create `pypi` and `testpypi` environments at repo Settings > Environments

## Keeping Docs Updated

When making changes that affect project setup, CLI interface, dependencies, project structure, or development workflows, update **README.md** and **CLAUDE.md** accordingly:

- **README.md** — user-facing: installation, usage examples, CLI options, AWS setup/quota instructions
- **CLAUDE.md** — agent-facing: project overview, tech stack, project structure, coding conventions
