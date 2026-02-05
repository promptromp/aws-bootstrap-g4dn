# aws-bootstrap-skill

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) plugin that provides a skill for provisioning and managing AWS EC2 GPU instances via the [aws-bootstrap](https://github.com/promptromp/aws-bootstrap-g4dn) CLI. It enables LLM coding agents to autonomously launch, monitor, and tear down GPU instances for ML/CUDA development workflows.

## Installation

### From GitHub

Register this repository as a plugin marketplace, then install:

```bash
# Add the marketplace (one-time setup)
/plugin marketplace add promptromp/aws-bootstrap-g4dn

# Install the plugin
/plugin install aws-bootstrap-skill@promptromp-aws-bootstrap-g4dn
```

### Local install (from repo checkout)

```bash
claude --plugin-dir ./aws-bootstrap-skill
```

### Prerequisites

The `aws-bootstrap` CLI must be installed separately:

```bash
pip install aws-bootstrap-g4dn
# or
uv pip install aws-bootstrap-g4dn
```

You also need:
- AWS credentials configured (`AWS_PROFILE` env var or `--profile` flag)
- An SSH key pair (default: `~/.ssh/id_ed25519`)

## Usage

Once the plugin is loaded, Claude Code will automatically use the `aws-bootstrap` skill when you ask about GPU instances. Examples:

- "Launch a GPU instance for me"
- "What GPU instances are running?"
- "Terminate all my GPU instances"
- "Show me available g5 instance types"
- "Clean up stale SSH entries and orphan EBS volumes"

The skill teaches Claude Code to use `--output json` for machine-readable output, handle spot pricing fallbacks, manage EBS data volumes, and clean up resources properly.

## What's Included

- **SKILL.md** -- Main skill with quick reference, common workflows, and error handling guidance
- **references/commands.md** -- Full command reference with all options, defaults, and JSON output schemas

## License

MIT -- see the [main repository](https://github.com/promptromp/aws-bootstrap-g4dn) for details.
