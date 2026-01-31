# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

aws-bootstrap-g4dn bootstraps AWS EC2 GPU instances (g4dn.xlarge default) running Deep Learning AMIs for hybrid local-remote development. It provisions cost-effective Spot Instances via AWS CLI with SSH key bootstrap for CUDA development.

Target workflows: Jupyter server-client, VSCode Remote SSH, and NVIDIA Nsight remote debugging.

## Tech Stack & Requirements

- **Python 3.14+** with **uv** package manager (astral-sh/uv)
- **AWS CLI v2** with a configured AWS profile (`AWS_PROFILE` env var)
- **direnv** for automatic venv activation (`.envrc` sources `.venv/bin/activate`)

## Development Setup

```bash
uv venv .venv
uv sync
direnv allow  # or manually: source .venv/bin/activate
```

## Project Status

Early-stage (v0.1.0) — project skeleton with no implementation code yet. The pyproject.toml has no dependencies declared. Implementation will likely involve boto3 for EC2 provisioning, instance bootstrap scripts, and infrastructure-as-code templates.
