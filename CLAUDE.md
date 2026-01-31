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
    ec2.py               # AMI lookup, security group, instance launch/find/terminate, polling
    ssh.py               # SSH key pair import, SSH readiness check, remote setup
    remote_setup.sh      # Uploaded & run on instance post-boot (GPU verify, Jupyter, etc.)
    tests/               # Unit tests (pytest)
```

Entry point: `aws-bootstrap = "aws_bootstrap.cli:main"` (installed via `uv sync`)

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

## Keeping Docs Updated

When making changes that affect project setup, CLI interface, dependencies, project structure, or development workflows, update **README.md** and **CLAUDE.md** accordingly:

- **README.md** — user-facing: installation, usage examples, CLI options, AWS setup/quota instructions
- **CLAUDE.md** — agent-facing: project overview, tech stack, project structure, coding conventions
