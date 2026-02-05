# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

aws-bootstrap-g4dn is a Python CLI tool (`aws-bootstrap`) that bootstraps AWS EC2 GPU instances (g4dn.xlarge default) running Deep Learning AMIs for hybrid local-remote development. It provisions cost-effective Spot Instances via boto3 with SSH key import, security group setup, instance polling, and remote setup automation.

Target workflows: Jupyter server-client, VSCode Remote SSH, and NVIDIA Nsight remote debugging.

## Tech Stack & Requirements

- **Python 3.12+** with **uv** package manager (astral-sh/uv) — used for venv creation, dependency management, and running the project
- **boto3** — AWS SDK for EC2 provisioning (AMI lookup, security groups, instance launch, waiters)
- **click** — CLI framework with built-in color support (`click.secho`, `click.style`)
- **pyyaml** — YAML serialization for `--output yaml`
- **tabulate** — Table formatting for `--output table`
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
    ec2.py               # AMI lookup, security group, instance launch/find/terminate, polling, spot pricing, EBS volume ops
    gpu.py               # GPU architecture mapping and GpuInfo dataclass
    output.py            # Output formatting: OutputFormat enum, emit(), echo/secho wrappers for structured output
    quota.py             # Service Quotas API: get/request GPU vCPU quotas (G/VT, P5, P4/P3/P2, DL families)
    ssh.py               # SSH key pair import, SSH readiness check, remote setup, ~/.ssh/config management, GPU queries, EBS mount
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
        test_output.py
        test_gpu.py
        test_ssh_config.py
        test_ssh_gpu.py
        test_ebs.py
        test_ssh_ebs.py
        test_quota.py
docs/
    nsight-remote-profiling.md # Nsight Compute, Nsight Systems, and Nsight VSCE remote profiling guide
    spot-request-lifecycle.md  # Research notes on spot request cleanup
aws-bootstrap-skill/             # Claude Code plugin
    .claude-plugin/
        plugin.json              # Plugin manifest (identity, metadata)
    skills/
        aws-bootstrap/
            SKILL.md             # Main skill definition (quick reference, workflows, error handling)
            references/
                commands.md      # Full command reference with options and JSON output schemas
    README.md                    # Plugin installation and usage
.claude-plugin/
    marketplace.json             # Marketplace discovery (points to aws-bootstrap-skill/)
```

Entry point: `aws-bootstrap = "aws_bootstrap.cli:main"` (installed via `uv sync`)

## CLI Commands

**Global option:** `--output` / `-o` controls output format: `text` (default, human-readable with color), `json`, `yaml`, `table`. Structured formats (json/yaml/table) suppress all progress messages and emit machine-readable output. Commands requiring confirmation (`terminate`, `cleanup`) require `--yes` in structured modes.

- **`launch`** — provisions an EC2 instance (spot by default, falls back to on-demand on capacity errors); adds SSH config alias (e.g. `aws-gpu1`) to `~/.ssh/config`; `--python-version` controls which Python `uv` installs in the remote venv; `--ssh-port` overrides the default SSH port (22) for security group ingress, connection checks, and SSH config; `--ebs-storage SIZE` creates and attaches a new gp3 EBS data volume (mounted at `/data`); `--ebs-volume-id ID` attaches an existing EBS volume (mutually exclusive with `--ebs-storage`)
- **`status`** — lists all non-terminated instances (including `shutting-down`) with type, IP, SSH alias, EBS data volumes, pricing (spot price/hr or on-demand), uptime, and estimated cost for running spot instances; `--gpu` flag queries GPU info via SSH, reporting both CUDA toolkit version (from `nvcc`) and driver-supported max (from `nvidia-smi`); `--instructions` (default: on) prints connection commands (SSH, Jupyter tunnel, VSCode Remote SSH, GPU benchmark) for each running instance; suppress with `--no-instructions`
- **`terminate`** — terminates instances by ID or SSH alias (e.g. `aws-gpu1`, resolved via `~/.ssh/config`), or all aws-bootstrap instances in the region if no arguments given; removes SSH config aliases; deletes associated EBS data volumes by default; `--keep-ebs` preserves volumes and prints reattach commands
- **`cleanup`** — removes stale `~/.ssh/config` entries for terminated/non-existent instances; compares managed SSH config blocks against live EC2 instances; `--include-ebs` also finds and deletes orphan EBS data volumes (volumes in `available` state whose linked instance no longer exists); `--dry-run` previews removals without modifying config; `--yes` skips the confirmation prompt
- **`list instance-types`** — lists EC2 instance types matching a family prefix (default: `g4dn`), showing vCPUs, memory, and GPU info
- **`list amis`** — lists available AMIs matching a name pattern (default: Deep Learning Base OSS Nvidia Driver GPU AMIs), sorted newest-first
- **`quota show`** — displays current spot and on-demand vCPU quota values via AWS Service Quotas API; `--family` filters by instance family (gvt, p5, p, dl; default: all)
- **`quota request`** — requests a quota increase; `--type` (spot/on-demand) and `--desired-value` are required; `--family` selects instance family (default: gvt); confirms in text mode, requires `--yes` in structured modes
- **`quota history`** — shows history of quota increase requests; optional `--family`, `--type`, and `--status` filters

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

## Structured Output Architecture

The `--output` option uses a context-aware suppression pattern via `aws_bootstrap/output.py`:

- **`output.echo()` / `output.secho()`** — wrap `click.echo`/`click.secho`; silent in non-text modes. Used in `ec2.py` and `ssh.py` for progress messages.
- **`is_text(ctx)`** — checks if the current output format is text. Used in `cli.py` to guard text-only blocks.
- **`emit(data, headers=..., ctx=...)`** — dispatches structured data to JSON/YAML/table renderers. No-op in text mode.
- **CLI helper guards** — `step()`, `info()`, `val()`, `success()`, `warn()` in `cli.py` check `is_text()` and return early in structured modes.
- Each CLI command builds a result dict alongside existing logic, emits it via `emit()` for non-text formats, and falls through to text output for text mode.
- **Confirmation prompts** (`terminate`, `cleanup`) require `--yes` in structured modes to avoid corrupting output.
- The spot-fallback `click.confirm()` in `ec2.py` auto-confirms in structured modes.

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

## EBS Data Volumes

The `--ebs-storage` and `--ebs-volume-id` options on `launch` create or attach persistent gp3 EBS volumes mounted at `/data`. The implementation spans three modules:

- **`ec2.py`** — Volume lifecycle: `create_ebs_volume`, `validate_ebs_volume`, `attach_ebs_volume`, `detach_ebs_volume`, `delete_ebs_volume`, `find_ebs_volumes_for_instance`. Constants `EBS_DEVICE_NAME` (`/dev/sdf`) and `EBS_MOUNT_POINT` (`/data`).
- **`ssh.py`** — `mount_ebs_volume()` SSHs to the instance and runs a shell script that detects the device, optionally formats it, mounts it, and adds an fstab entry.
- **`cli.py`** — Orchestrates the flow: create/validate → attach → wait for SSH → mount. Mount failures are non-fatal (warn and continue).

### Tagging strategy

Volumes are tagged for discovery by `status` and `terminate`:

| Tag | Value | Purpose |
|-----|-------|---------|
| `created-by` | `aws-bootstrap-g4dn` | Standard tool-managed resource tag |
| `Name` | `aws-bootstrap-data-{instance_id}` | Human-readable in AWS console |
| `aws-bootstrap-instance` | `i-xxxxxxxxx` | Links volume to instance for `find_ebs_volumes_for_instance` |

### NVMe device detection

On Nitro instances (g4dn), `/dev/sdf` is remapped to `/dev/nvmeXn1`. The mount script detects the correct device by matching the volume ID serial number via `lsblk -o NAME,SERIAL -dpn`, with fallbacks to `/dev/nvme1n1`, `/dev/xvdf`, `/dev/sdf`.

### Spot interruption and terminate cleanup

Non-root EBS volumes attached via API have `DeleteOnTermination=False` by default. This means data volumes **survive spot interruptions** — when AWS reclaims the instance, the volume detaches and becomes `available`, preserving all data. The user can reattach it to a new instance with `--ebs-volume-id`.

The `terminate` command discovers volumes via `find_ebs_volumes_for_instance`, waits for them to detach (becomes `available`), then deletes them. `--keep-ebs` skips deletion and prints the volume ID with a reattach command.

## Agent Skill (Claude Code Plugin)

The `aws-bootstrap-skill/` directory contains a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) plugin following the [Agent Skills](https://agentskills.io/) standard. It enables LLM coding agents to autonomously provision, manage, and tear down AWS GPU instances via the `aws-bootstrap` CLI.

- **`.claude-plugin/marketplace.json`** (repo root) — marketplace discovery metadata, points to `aws-bootstrap-skill/`
- **`aws-bootstrap-skill/.claude-plugin/plugin.json`** — plugin manifest (identity, metadata, keywords)
- **`aws-bootstrap-skill/skills/aws-bootstrap/SKILL.md`** — main skill definition (~200 lines): prerequisites, quick reference, structured output guidance, common workflows, remote instance environment (~/venv, /data), error handling
- **`aws-bootstrap-skill/skills/aws-bootstrap/references/commands.md`** — full command reference (~280 lines): all options with defaults and JSON output schemas for every command
- **`aws-bootstrap-skill/README.md`** — plugin installation and usage instructions

The skill uses progressive disclosure: SKILL.md is loaded as the quick reference, `references/commands.md` is loaded on demand for detailed option docs. The skill instructs agents to use `--output json` for machine-readable output and documents the pre-installed remote environment (`~/venv` with CUDA-matched PyTorch, `/data` EBS mount for datasets).

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
