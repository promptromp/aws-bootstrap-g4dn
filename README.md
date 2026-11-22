# aws-bootstrap-g4dn

--------------------------------------------------------------------------------

[![CI](https://github.com/promptromp/aws-bootstrap-g4dn/actions/workflows/ci.yml/badge.svg)](https://github.com/promptromp/aws-bootstrap-g4dn/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/promptromp/aws-bootstrap-g4dn)](https://github.com/promptromp/aws-bootstrap-g4dn/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/aws-bootstrap-g4dn)](https://pypi.org/project/aws-bootstrap-g4dn/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aws-bootstrap-g4dn)](https://pypi.org/project/aws-bootstrap-g4dn/)

One command to go from zero to a **fully configured GPU dev box** on AWS — with CUDA-matched PyTorch, Jupyter, SSH aliases, and a GPU benchmark ready to run.

```bash
aws-bootstrap launch          # Spot g4dn.xlarge in ~3 minutes
ssh aws-gpu1                  # You're in, venv activated, PyTorch works
```

### ✨ Key Features

| | Feature | Details |
|---|---|---|
| 🚀 | **One-command launch** | Spot (default) or on-demand, with automatic fallback on capacity errors |
| 🔑 | **Auto SSH config** | Adds `aws-gpu1` alias to `~/.ssh/config` — no IP juggling. Cleaned up on terminate |
| 🐍 | **CUDA-aware PyTorch** | Detects the installed CUDA toolkit (`nvcc`) and installs PyTorch from the matching wheel index — no more `torch.version.cuda` mismatches |
| ✅ | **PyTorch smoke test** | Runs a quick `torch.cuda` matmul after setup to verify the GPU stack works end-to-end |
| 📊 | **GPU benchmark included** | CNN (MNIST) + Transformer benchmarks with FP16/FP32/BF16 precision and tqdm progress |
| 📓 | **Jupyter ready** | Lab server auto-starts as a systemd service on port 8888 — just SSH tunnel and open |
| 🖥️ | **`status --gpu`** | Shows CUDA toolkit version, driver max, GPU architecture, spot pricing, uptime, and estimated cost |
| 🌍 | **Multi-region status** | `status` with no `--region` finds instances across every enabled region and labels each with its region |
| 💾 | **EBS data volumes** | Attach persistent storage at `/data` — survives spot interruptions and termination, reattach to new instances |
| 🗑️ | **Clean terminate** | Stops instances, removes SSH aliases, cleans up EBS volumes (or preserves with `--keep-ebs`) |
| 🤖 | **[Agent Skill](https://agentskills.io/)** | Included Claude Code plugin lets LLM agents autonomously provision, manage, and tear down GPU instances |

### 🎯 Target Workflows

1. **Jupyter server-client** — Jupyter runs on the instance, connect from your local browser
2. **VSCode Remote SSH** — opens `~/workspace` with pre-configured CUDA debug/build tasks and an example `.cu` file
3. **NVIDIA Nsight remote debugging** — GPU debugging over SSH

---

## Requirements

1. AWS profile configured with relevant permissions (profile name can be passed via `--profile` or read from `AWS_PROFILE` env var)
2. AWS CLI v2 — see [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
3. Python 3.12+ and [uv](https://github.com/astral-sh/uv)
4. An SSH key pair (see below)

## Installation

### From PyPI

```bash
pip install aws-bootstrap-g4dn
```

### With uvx (no install needed)

[uvx](https://docs.astral.sh/uv/guides/tools/) runs the CLI directly in a temporary environment — no global install required:

```bash
uvx --from aws-bootstrap-g4dn aws-bootstrap launch
uvx --from aws-bootstrap-g4dn aws-bootstrap status
uvx --from aws-bootstrap-g4dn aws-bootstrap terminate
```

### From source (development)

```bash
git clone https://github.com/promptromp/aws-bootstrap-g4dn.git
cd aws-bootstrap-g4dn
uv venv
uv sync
```

All methods install the `aws-bootstrap` CLI.

For development, `pre-commit run --all` runs lint, format, type checks and the tests your
change can reach ([pytest-impacted](https://github.com/promptromp/pytest-impacted)); `uv run pytest`
runs the whole suite, as CI does on every push to `main`.

#### Optional: auto-activate the venv with direnv

A sample [direnv](https://direnv.net/) config is provided at [`.envrc.example`](.envrc.example). It activates the project venv (and optionally sets `AWS_PROFILE`) automatically when you `cd` into the repo:

```bash
cp .envrc.example .envrc
# edit .envrc to uncomment/set AWS_PROFILE if desired
direnv allow
```

`.envrc` is git-ignored, so your local copy stays out of version control.

## SSH Key Setup

The CLI expects an Ed25519 SSH public key at `~/.ssh/id_ed25519.pub` by default. If you don't have one, generate it:

```bash
ssh-keygen -t ed25519
```

Accept the default path (`~/.ssh/id_ed25519`) and optionally set a passphrase. The key pair is imported into AWS automatically on first launch.

To use a different key, pass `--key-path`:

```bash
aws-bootstrap launch --key-path ~/.ssh/my_other_key.pub
```

**Robust key handling** (so you never end up with an instance you can't reach):

- **Missing local key** — if `--key-path` doesn't exist, `launch` auto-generates an Ed25519 key pair there (instead of aborting).
- **Name collision with a different key** — if an AWS key pair already exists with the target `--key-name` but its public key differs from your local key (e.g. created from another machine), the existing AWS key pair is **left untouched** and your local key is imported under a deterministic derived name `aws-bootstrap-key-<fp8>`, which the instance is launched with. You always hold the matching private key.
- **Unreachable instance** — if SSH still fails with an authentication/host-key error, `launch` **stops immediately** and prints the real `ssh` error (no more silent 5-minute "SSH not ready" loop masking a `Permission denied (publickey)`).
- **Passphrase-protected key** — the CLI connects with `BatchMode=yes` (it never prompts). So
  `launch` / `cluster launch` (including `--dry-run`) check, before creating anything, that the key
  works without a prompt: either the agent ssh would use holds it (it honours `IdentityAgent`, e.g.
  1Password, even with no private key file on disk), or the key has no passphrase. If neither, they
  stop *before* creating a billed instance setup could never reach, and print the fix
  (`ssh-add --apple-use-keychain <key>` on macOS, `ssh-add <key>` elsewhere).
  On macOS, with `UseKeychain yes` in `~/.ssh/config`, the Keychain may supply the passphrase; that
  can't be checked, so the CLI only warns. `--skip-key-check` turns the check off.

## Usage

### 🚀 Launching an Instance

```bash
# Show available commands
aws-bootstrap --help

# Dry run — validates AMI lookup, key import, and security group without launching
aws-bootstrap launch --dry-run

# Launch a spot g4dn.xlarge (default)
aws-bootstrap launch

# Launch on-demand in a specific region with a custom instance type
aws-bootstrap launch --on-demand --instance-type g5.xlarge --region us-east-1

# Try multiple regions in order until one has spot capacity
aws-bootstrap launch --region us-west-2 --region us-east-1 --region eu-west-1

# Keep retrying (bounded exponential backoff) until spot capacity frees up
aws-bootstrap launch --wait --wait-timeout 30m
aws-bootstrap launch --region us-west-2 --region us-east-1 --wait --wait-timeout 1h

# Launch without running the remote setup script
aws-bootstrap launch --no-setup

# Use a specific Python version in the remote venv
aws-bootstrap launch --python-version 3.13

# Use a non-default SSH port
aws-bootstrap launch --ssh-port 2222

# Attach a persistent EBS data volume (96 GB gp3, mounted at /data)
aws-bootstrap launch --ebs-storage 96

# Reattach an existing EBS volume from a previous instance (the volume decides the region and AZ)
aws-bootstrap launch --ebs-volume-id vol-0abc123def456

# Give the instance AWS credentials through an existing IAM instance profile (e.g. to read S3)
aws-bootstrap launch --iam-instance-profile my-ec2-s3-profile

# Spot prices differ a lot between AZs: try each region's AZs cheapest-first
aws-bootstrap launch --instance-type g6e.2xlarge --cheapest-az --region us-west-2 --region us-east-1

# Or pick the AZ yourself (implies its region; repeatable, tried in order)
aws-bootstrap launch --availability-zone us-west-2b --ebs-storage 200

# Use a specific AWS profile
aws-bootstrap launch --profile my-aws-profile
```

After launch, the CLI:

1. **Creates/attaches EBS volume** (if `--ebs-storage` or `--ebs-volume-id` was specified)
2. **Adds an SSH alias** (e.g. `aws-gpu1`) to `~/.ssh/config`
3. **Runs remote setup** — installs utilities, creates a Python venv, installs CUDA-matched PyTorch, sets up Jupyter
4. **Mounts EBS volume** at `/data` (if applicable — formats new volumes, mounts existing ones as-is)
5. **Runs a CUDA smoke test** — verifies `torch.cuda.is_available()` and runs a quick GPU matmul
6. **Prints connection commands** — SSH, Jupyter tunnel, GPU benchmark, and terminate

```bash
ssh aws-gpu1                  # venv auto-activates on login
```

### 🌍 Finding Capacity (regions & `--wait`)

Spot `InsufficientInstanceCapacity` is scoped to a **region and availability zone** — a type that's unavailable in `us-west-2` right now may be plentiful in `us-east-1`, and capacity for a given AZ frees up continuously as other instances terminate. Two options help you get a GPU without babysitting the prompt:

- **Multiple regions** — pass `--region` more than once. Each launch attempt tries the regions **in the order given**, spot-first, and uses the first one with capacity:

  ```bash
  aws-bootstrap launch --region us-west-2 --region us-east-1 --region eu-west-1
  ```

- **`--wait`** — on insufficient spot capacity, keep retrying with **capped, jittered exponential backoff** until `--wait-timeout` (default `30m`; accepts `90s`, `30m`, `1h`, or bare seconds). On timeout it **hard-fails** (it does not silently fall back to on-demand):

  ```bash
  aws-bootstrap launch --region us-west-2 --region us-east-1 --wait --wait-timeout 1h
  ```

**How `--wait` + multiple `--region` combine:** a **region sweep is the inner loop, backoff is the outer loop**. Each cycle tries spot in every `--region` in order *with no delay between regions*; only when **all** regions miss does it sleep (backoff) and sweep again. So `--wait --region A --region B` means "try A then B instantly; if both dry, back off and retry A then B" — repeating until timeout — *not* "wait on A, then try B." Backoff escalates per sweep (not per region), region order wins every tie, and `--wait-timeout` is total wall-clock. See [docs/capacity-and-retry.md](docs/capacity-and-retry.md#how---wait-and-multiple---region-combine) for the full model.

Quota errors (`VcpuLimitExceeded`, `MaxSpotInstanceCountExceeded`) and `SpotMaxPriceTooLow` are **never retried by `--wait`** (waiting can't fix them). In multi-region mode they are *not* fatal on their own: the launcher prints a `WARNING` for that region (with a region-pinned `aws-bootstrap quota …` hint), skips it, and tries the next `--region` — failing hard only once **every** region is blocked, with an aggregated message listing each region's reason and hint. Without `--wait`, a fully-exhausted spot pass still offers the interactive on-demand fallback (across all regions).

**Region default precedence** (a behavior change — previously hardcoded to `us-west-2`): explicit `--region` flags → `AWS_DEFAULT_REGION` / active profile region → `us-west-2`. This applies to every command, so a profile configured for `us-east-1` now operates in `us-east-1` by default. The active region is shown in command output.

#### Choosing the availability zone (`--cheapest-az`, `--availability-zone`)

By default EC2 picks the AZ within each region, but **spot prices can differ several-fold between AZs
of the same region** (e.g. g6e.2xlarge: us-west-2b $0.99/h vs $2.21/h in us-west-2a/c/d). The AZ also
matters with `--ebs-storage`: the new volume is created in the instance's AZ, and every later
`--ebs-volume-id` reattach is pinned there.

- **`--cheapest-az`** (spot only) ranks each region's AZs that offer the type (and have a default
  subnet) by **current spot price** and tries them cheapest-first. A capacity miss (or a spot price
  over the maximum, or an AZ that doesn't support the launch) moves to the next AZ in the same region
  before the next region. Every region's plan is checked before anything launches.
- **`--availability-zone AZ`** / **`--az AZ`** (repeatable) restricts the launch to those AZs, tried in
  the order given. An AZ implies its region: without `--region`, the regions are the AZs' own. With
  `--cheapest-az` too, only the listed AZs are ranked.
- The chosen AZ and its spot price are always printed (`spot_price_per_hour` in structured output),
  and `--dry-run` shows the AZ plan for every `--region`.

Price semantics, stated plainly:
- **Region order still comes first.** Ranking happens *within* a region, so with `--region us-west-2
  --region us-east-1` every us-west-2 AZ (even a $2.21 one) is tried before us-east-1's cheapest.
- **A capacity miss can cost you.** Falling through from the cheapest AZ lands in the next one, which
  may be much pricier (and, with `--ebs-storage`, pins the volume there). For *the cheap AZ or
  nothing*, pin it and wait: `--availability-zone us-west-2b --wait`.
- Both flags are incompatible with `--ebs-volume-id` (the volume already decides the AZ);
  `--cheapest-az` is incompatible with `--on-demand` (on-demand prices don't vary by AZ).

See [docs/capacity-and-retry.md](docs/capacity-and-retry.md) for the backoff design and recommended region lists per instance family.

### 🔑 AWS access from the instance (`--iam-instance-profile`)

Code on the instance that calls AWS (reading a dataset from S3, pulling from ECR, writing
checkpoints) needs credentials. Don't copy access keys onto it: give it an IAM **instance
profile** instead, and the AWS CLI and SDKs there pick up short-lived credentials on their own.

```bash
aws-bootstrap launch --iam-instance-profile my-ec2-s3-profile          # name or ARN
aws-bootstrap cluster launch --cluster-id ml1 --nodes 2 --iam-instance-profile my-ec2-s3-profile
```

- aws-bootstrap only **attaches** a profile you already have; it never creates roles, policies or
  profiles. Create one with a role whose policy grants what the instance needs (e.g. `s3:GetObject`
  on your bucket), or ask whoever manages IAM for your account.
- The profile is checked before anything is created: a name that doesn't exist stops the launch.
  If you lack `iam:GetInstanceProfile`, the check only warns (EC2 still rejects a bad profile before
  launching anything). Launching with a profile needs `iam:PassRole` on its role.
- `status` shows each instance's profile (`iam_instance_profile` in structured output).
- Growing a cluster whose nodes have a different profile warns and names them, since a job that
  needs the credentials on every rank would fail on the others.
- For an instance that's already running, use the AWS CLI:
  ```bash
  aws ec2 associate-iam-instance-profile --region us-west-2 --instance-id i-0abc... \
      --iam-instance-profile Name=my-ec2-s3-profile
  # to swap one: describe-iam-instance-profile-associations, then replace-iam-instance-profile-association
  ```

### 🔧 What Remote Setup Does

The setup script runs automatically on the instance after SSH becomes available:

| Step | What |
|------|------|
| **GPU verify** | Confirms `nvidia-smi` and `nvcc` are working |
| **Utilities** | Installs `htop`, `tmux`, `tree`, `jq`, `ffmpeg` |
| **Python venv** | Creates `~/venv` with `uv`, auto-activates in `~/.bashrc`. Use `--python-version` to pin a specific Python (e.g. `3.13`) |
| **CUDA-aware PyTorch** | Detects the CUDA toolkit version → installs PyTorch from the newest `cu{TAG}` wheel index PyTorch publishes at or below it (discovered from the index, not a hardcoded list) |
| **CUDA smoke test** | Runs `torch.cuda.is_available()` + GPU matmul to verify the stack |
| **GPU benchmark** | Copies `gpu_benchmark.py` to `~/gpu_benchmark.py` |
| **GPU smoke test notebook** | Copies `gpu_smoke_test.ipynb` to `~/gpu_smoke_test.ipynb` (open in JupyterLab) |
| **Jupyter** | Configures and starts JupyterLab as a systemd service on port 8888 |
| **SSH keepalive** | Configures server-side keepalive to prevent idle disconnects |
| **VSCode workspace** | Creates `~/workspace/.vscode/` with `launch.json` and `tasks.json` (auto-detected `cuda-gdb` path and GPU arch), plus an example `saxpy.cu` |

### 🧩 Packages that compile against the instance's PyTorch

Some packages (flash-attn, causal-conv1d, mamba-ssm, …) build CUDA kernels against the installed
torch, so they need `--no-build-isolation` in `~/venv`, and the build can take many minutes.
[`examples/compiled-extensions/install_compiled.sh`](examples/compiled-extensions/) does that, with
an optional S3 wheel cache so the next instance installs in seconds:

```bash
ssh aws-gpu1 'WHEEL_CACHE=s3://my-bucket/wheels bash -s -- flash-attn==2.8.3' \
    < examples/compiled-extensions/install_compiled.sh
```

A wheel's filename doesn't record what it was built against, so cached wheels are kept under a key
that does: torch version, torch's CUDA, the nvcc toolkit, Python, CPU architecture, libc and GPU
generation. pip decides whether a cached wheel satisfies each requested version. The instance
needs S3 access for the cache (`--iam-instance-profile`).

### 📊 GPU Benchmark

A GPU throughput benchmark is pre-installed at `~/gpu_benchmark.py` on every instance:

```bash
# Run both CNN and Transformer benchmarks (default)
ssh aws-gpu1 '~/venv/bin/python ~/gpu_benchmark.py'

# CNN only, quick run
ssh aws-gpu1 '~/venv/bin/python ~/gpu_benchmark.py --mode cnn --benchmark-batches 20'

# Transformer only with custom batch size
ssh aws-gpu1 '~/venv/bin/python ~/gpu_benchmark.py --mode transformer --transformer-batch-size 16'

# Run CUDA diagnostics first (tests FP16/FP32 matmul, autocast, etc.)
ssh aws-gpu1 '~/venv/bin/python ~/gpu_benchmark.py --diagnose'

# Force FP32 precision (if FP16 has issues on your GPU)
ssh aws-gpu1 '~/venv/bin/python ~/gpu_benchmark.py --precision fp32'
```

Reports: iterations/sec, samples/sec, peak GPU memory, and avg batch time for each model.

### 📓 Jupyter (via SSH Tunnel)

```bash
ssh -NL 8888:localhost:8888 aws-gpu1
# Then open: http://localhost:8888
```

Or with explicit key/IP:
```bash
ssh -i ~/.ssh/id_ed25519 -NL 8888:localhost:8888 ubuntu@<public-ip>
```

A **GPU smoke test notebook** (`~/gpu_smoke_test.ipynb`) is pre-installed on every instance. Open it in JupyterLab to interactively verify the CUDA stack, run FP32/FP16 matmuls, train a small CNN on MNIST, and visualise training loss and GPU memory usage.

### 🖥️ VSCode Remote SSH

The remote setup creates a `~/workspace` folder with pre-configured CUDA debug and build tasks:

```
~/workspace/
├── .vscode/
│   ├── launch.json   # CUDA debug configs (cuda-gdb path auto-detected)
│   └── tasks.json    # nvcc build tasks (GPU arch auto-detected, e.g. sm_75)
└── saxpy.cu          # Example CUDA source — open and press F5 to debug
```

Connect directly from your terminal:

```bash
code --folder-uri vscode-remote://ssh-remote+aws-gpu1/home/ubuntu/workspace
```

Then install the [Nsight VSCE extension](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) on the remote when prompted. Open `saxpy.cu`, set a breakpoint, and press F5.

See [Nsight remote profiling guide](docs/nsight-remote-profiling.md) for more details on CUDA debugging and profiling workflows.

### 📤 Structured Output

All commands support `--output` / `-o` for machine-readable output — useful for scripting, piping to `jq`, or LLM tool-use:

```bash
# JSON output (pipe to jq)
aws-bootstrap -o json status
aws-bootstrap -o json status | jq '.instances[0].instance_id'

# YAML output
aws-bootstrap -o yaml status

# Table output
aws-bootstrap -o table status

# Works with all commands
aws-bootstrap -o json list instance-types | jq '.[].instance_type'
aws-bootstrap -o json launch --dry-run
aws-bootstrap -o json terminate --yes
aws-bootstrap -o json cleanup --dry-run
```

Supported formats: `text` (default, human-readable with color), `json`, `yaml`, `table`. Commands that require confirmation (`terminate`, `cleanup`) require `--yes` in structured output modes.

### 📋 Listing Resources

```bash
# List all g4dn instance types (default)
aws-bootstrap list instance-types

# List a different instance family
aws-bootstrap list instance-types --prefix g6

# --prefix is a *prefix* match, so it also finds hyphenated families:
#   p6  -> p6-b200.48xlarge, p6-b300.48xlarge
#   g6  -> g6, g6e and g6f types (fractional-GPU types such as g6f show their
#          share of one GPU, e.g. "1/8 L4")
aws-bootstrap list instance-types --prefix p6

# List Deep Learning Base AMIs — every OS (Ubuntu 22.04/24.04/26.04, Amazon Linux
# 2023, …) and architecture; each AMI is labelled with its region
aws-bootstrap list amis

# List AMIs with a custom filter
aws-bootstrap list amis --filter "ubuntu/images/hvm-ssd-gp3/ubuntu-noble*"

# --arch narrows to one architecture, e.g. the ARM64 builds that arm64 GPU types
# (g5g) run — see "ARM64 instance types" below
aws-bootstrap list amis --arch arm64

# Use a specific region (the active region is shown in the output header)
aws-bootstrap list instance-types --region us-east-1
aws-bootstrap list amis --region us-east-1

# --region is repeatable (-r for short) to compare across regions
aws-bootstrap list amis -r us-east-1 -r us-west-2
aws-bootstrap list instance-types --prefix g5 -r us-east-1 -r eu-west-1
```

`list instance-types` shows a **Quota Family** column (`gvt`/`p`/`dl`) — the AWS
vCPU quota family each type draws from. These group multiple prefixes (e.g. all
G/VT types, including `g5`, share `gvt`), so the suggested `--family` may not
look like your `--prefix`. The output then ends with copy-paste **Next steps**
for that family — a `quota show` and a `quota request` command pinned to the
queried region — so you can go straight from "is this type available?" to
checking and raising your vCPU quota.

#### ARM64 instance types

An AMI must match its instance type's CPU architecture, and not every GPU family
is x86_64 — `g5g` (NVIDIA T4g) is arm64-only. `launch` reads the architecture
from the instance type (via EC2) and looks up an AMI for it. AWS names the
Deep Learning Base AMI per architecture (`Deep Learning Base OSS …` for x86_64,
`Deep Learning ARM64 Base OSS …` for arm64), and the default `--ami-filter`
(`Deep Learning*Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*`) spans both,
so no extra flag is needed — it resolves to the same image as AWS's public SSM
"latest DLAMI" parameter for that architecture:

```bash
aws-bootstrap launch --instance-type g5g.xlarge
```

If you pass your own `--ami-filter` and it matches no image for the required
architecture, the error says so and shows how to list what the region offers,
rather than a generic "no AMI found".

Ubuntu 26.04 builds of the same AMI are also published; to use one, pass
`--ami-filter "Deep Learning*Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)*"`.

### 🖥️ Managing Instances

```bash
# Show all aws-bootstrap instances across every enabled region (including shutting-down).
# Each instance is labelled with its region. Also reports orphaned EBS data
# volumes (detached but still billed) with cost estimates and cleanup hints.
aws-bootstrap status

# Include GPU info (CUDA toolkit + driver version, GPU name, architecture) via SSH
aws-bootstrap status --gpu

# Hide connection commands (shown by default for each running instance)
aws-bootstrap status --no-instructions

# Restrict the query to one region
aws-bootstrap status --region us-east-1

# Restrict to several regions (--region is repeatable, -r for short)
aws-bootstrap status --region us-east-1 --region us-west-2
aws-bootstrap status -r us-east-1 -r eu-west-1

# Terminate all aws-bootstrap instances (with confirmation prompt)
aws-bootstrap terminate

# Terminate but preserve EBS data volumes for reuse
aws-bootstrap terminate --keep-ebs

# Terminate by SSH alias (resolved via ~/.ssh/config)
aws-bootstrap terminate aws-gpu1

# Terminate by instance ID
aws-bootstrap terminate i-abc123

# Mix aliases and instance IDs
aws-bootstrap terminate aws-gpu1 i-def456

# Skip confirmation prompt
aws-bootstrap terminate --yes

# Named instances are checked for the created-by tag first, so a wrong ID (or the
# wrong --region) can't destroy an unrelated instance. --force overrides the check.
aws-bootstrap terminate i-notmine        # -> refuses, explains why
aws-bootstrap terminate i-notmine --force

# Sync ~/.ssh/config with live instances across ALL regions
# (removes stale aliases left by spot reclaims / out-of-band terminations).
aws-bootstrap cleanup

# Preview (diagnostic) — shows stale / missing / drifted, changes nothing
aws-bootstrap cleanup --dry-run

# --sync also ADDS aliases for live instances missing one, and REPAIRS
# aliases whose IP drifted (e.g. after a stop/start)
aws-bootstrap cleanup --sync

# Also find and delete orphan EBS data volumes
aws-bootstrap cleanup --include-ebs --dry-run
aws-bootstrap cleanup --include-ebs --yes
```

`cleanup` scans **all enabled regions** (an SSH alias records no region, so staleness can only be judged globally). If a region query fails it reports the stale aliases but does **not** remove them — it can't prove the instance isn't alive in the region it couldn't reach. It never terminates instances.

`status --gpu` reports both the **installed CUDA toolkit** version (from `nvcc`) and the **maximum CUDA version supported by the driver** (from `nvidia-smi`), so you can see at a glance whether they match:

```
CUDA: 12.8 (driver supports up to 13.0)
```

SSH aliases are managed automatically — they're created on `launch`, shown in `status`, and cleaned up on `terminate`. Aliases use sequential numbering (`aws-gpu1`, `aws-gpu2`, etc.) and never reuse numbers from previous instances. You can use aliases anywhere you'd use an instance ID, e.g. `aws-bootstrap terminate aws-gpu1`.

## EBS Data Volumes

Attach persistent EBS storage to keep datasets and model checkpoints across instance lifecycles. Volumes are mounted at `/data` and persist independently of the instance.

```bash
# Create a new 96 GB gp3 volume, formatted and mounted at /data
aws-bootstrap launch --ebs-storage 96

# After terminating with --keep-ebs, reattach the same volume to a new instance
aws-bootstrap terminate --keep-ebs
# Output: Preserving EBS volume: vol-0abc123...
#         Reattach with: aws-bootstrap launch --ebs-volume-id vol-0abc123...

aws-bootstrap launch --ebs-volume-id vol-0abc123def456
```

Key behaviors:
- `--ebs-storage` and `--ebs-volume-id` are mutually exclusive
- New volumes are formatted as ext4; existing volumes are mounted as-is
- Volumes are tagged for automatic discovery by `status` and `terminate`
- `terminate` deletes data volumes by default; use `--keep-ebs` to preserve them
- **Ownership guard** — `terminate` with explicit instance IDs/aliases verifies each target carries the `created-by=aws-bootstrap-g4dn` tag and refuses (without terminating anything) if one doesn't; `--force` overrides. Terminating *all* instances already selects by tag, so it skips the check
- **Orphan visibility** — `status` always reports orphaned data volumes (detached, linked instance gone) with size, region, and an estimated monthly cost, plus reattach/delete commands — even when you have zero running instances. Orphans bill silently (~$0.08/GB-month for gp3), so this is your safety net after a spot reclaim or a forgotten `--keep-ebs`
- **Orphan cleanup** — use `aws-bootstrap cleanup --include-ebs` to find and delete orphan volumes (e.g. from spot interruptions or forgotten `--keep-ebs` volumes). Use `--dry-run` to preview
- **Spot-safe** — data volumes survive spot interruptions. If AWS reclaims your instance, the volume detaches automatically and can be reattached to a new instance with `--ebs-volume-id`
- **The volume decides the region and AZ** — EBS volumes are tied to a single availability zone, and an instance can only attach a volume in its own AZ. `--ebs-volume-id` finds the volume first (in your `--region`s, or the default region, then every enabled region) and launches there, pinned to its AZ, so you never hit a "wrong AZ" attach failure and don't need `--region`. A volume outside an explicit `--region` is an error that names where it actually is. (One consequence: spot capacity is then constrained to that single AZ, so a launch may need `--wait` to ride out a temporary shortage.)
- **Checked before launch** — a volume that isn't `available` (still attached, e.g. its old instance hasn't finished terminating) stops the launch before an instance is created, naming the instance it's attached to.
- Mount failures are non-fatal — the instance remains usable

## Multi-node training clusters (preview)

Launch several GPU instances as a coordinated **cluster** for multi-node distributed training (e.g. PyTorch DDP via `torchrun`). A cluster is identified by an arbitrary `--cluster-id` used as an EC2 tag — AWS tags are the source of truth, so there's no local state file to keep in sync.

> **Full walkthrough:** [docs/multi-node-training.md](docs/multi-node-training.md) takes you from zero to a running multi-node training job, with runnable example scripts in [examples/cluster/](examples/cluster/) (a DDP training script and a data-prep script).

```bash
# Launch a 4-node cluster (single AZ, cluster placement group, shared SG)
aws-bootstrap cluster launch --cluster-id ml1 --nodes 4 --instance-type g5.xlarge --region us-east-1

# Re-run with a higher --nodes to grow the cluster incrementally
aws-bootstrap cluster launch --cluster-id ml1 --nodes 6

# See the cluster's nodes (rank, state, AZ, IP); omit --cluster-id to list all clusters
aws-bootstrap cluster status --cluster-id ml1

# Verify the cluster (reachability, GPU, consistent CUDA) + run a distributed canary
aws-bootstrap cluster prepare --cluster-id ml1

# Re-run the canary any time as a heartbeat
aws-bootstrap cluster test --cluster-id ml1

# Run your own training script across all nodes (torchrun, c10d rendezvous).
# Args after `--` are passed to your script; per-node logs land under --log-dir.
aws-bootstrap cluster run --cluster-id ml1 train.py -- --epochs 5

# ...with a data-prep step that runs on each node first (e.g. pull from S3 to /data)
aws-bootstrap cluster run --cluster-id ml1 --data-script prep.sh train.py -- --epochs 5

# Tear it all down (nodes, SSH aliases, placement group)
aws-bootstrap cluster terminate --cluster-id ml1 --yes
```

Key behaviors:
- **One AZ + cluster placement group** — all nodes land in the same availability zone inside a cluster placement group for low-latency NCCL communication. Incremental adds reuse that AZ. For a **new** cluster, `--availability-zone AZ` picks that AZ and `--cheapest-az` (spot) tries the region's AZs cheapest-first for the first node; the rest follow it. An existing cluster stays in its own AZ.
- **Shared security group** — a self-referencing ingress rule lets cluster members reach each other on the NCCL/rendezvous ports (in addition to SSH).
- **Per-node SSH aliases** — each node gets an `aws-<cluster-id>-<rank>` alias in `~/.ssh/config` (e.g. `ssh aws-ml1-0`).
- **Stable ranks** — nodes are tagged with a stable rank (`0..N-1`); rank 0 is the rendezvous/master node (it also trains).
- **Verify before you train** — `cluster prepare` checks every node is reachable, has a GPU, and runs a **consistent** CUDA version (it fails fast on version skew), writes a per-node cluster config, then runs a built-in distributed **canary** (a tiny DDP all-reduce + a few SGD steps across all nodes) to prove NCCL/rendezvous works end-to-end. `cluster test` re-runs that canary on demand.
- **`cluster run` — distributed training** — distributes your `SCRIPT` to every node and runs it across the cluster. A `.py` script is launched with `torchrun` (c10d rendezvous, all nodes pointed at rank 0); a `.sh` script runs as-is with the `AWSB_*` environment contract exported (`AWSB_NODE_RANK`, `AWSB_NUM_NODES`, `AWSB_NUM_GPUS_PER_NODE`, `AWSB_NODE_IPS`, `AWSB_MASTER_ADDR`) so you can wire up any launcher. Per-node stdout/stderr are saved under `--log-dir`.
- **Data-prep convention** — pass `--data-script prep.sh` and it's copied to every node and run **once, before training** (it's a barrier — training won't start until all nodes finish prep). Only the script's **exit code** is the success signal, so start it with `set -euo pipefail` to ensure a failed step actually fails the prep. The recommended pattern is an idempotent S3 pull into the per-node `/data` EBS mount (launch the cluster with `--iam-instance-profile` so the nodes can read the bucket):
  ```bash
  # prep.sh
  set -euo pipefail
  test -f /data/.prepared && exit 0
  aws s3 sync s3://my-bucket/dataset /data/dataset
  touch /data/.prepared
  ```
- **EFA / AZ caveat** — `g4dn`/`g5` instances run NCCL over ordinary VPC networking (EFA is essentially P-series only), which is fine for dev/learning/modest scale. Pinning to one AZ means a spot shortage there affects the whole cluster.

## EC2 vCPU Quotas

AWS accounts have [service quotas](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) that limit how many vCPUs you can run per instance family. New or lightly-used accounts often have a **default quota of 0 vCPUs** for GPU instance families (G and VT), which will cause errors on launch:

- **Spot**: `MaxSpotInstanceCountExceeded`
- **On-Demand**: `VcpuLimitExceeded`

Check your current quotas (g4dn.xlarge requires at least 4 vCPUs):

```bash
# Built-in: show all GPU family quotas
aws-bootstrap quota show

# Show only G/VT family quotas
aws-bootstrap quota show --family gvt

# Show P family quotas (P2 through P6)
aws-bootstrap quota show --family p

# The active region is shown in the output header. --region is repeatable
# (-r) to compare quotas across regions:
aws-bootstrap quota show --family gvt -r us-east-1 -r us-west-2

# Or use the AWS CLI directly:
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-3819A6DF \
  --region us-west-2
```

Request increases:

```bash
# `aws-bootstrap quota show` prints a ready-to-run `quota request` command
# with a --desired-value above your current quota and pinned to --region.
# The desired value must EXCEED the current quota (AWS rejects <= current),
# so pick a value accordingly (8 shown as an example):
aws-bootstrap quota show --family gvt --region us-west-2
aws-bootstrap quota request --type spot --desired-value 8 --region us-west-2

# Request a P family spot quota increase
aws-bootstrap quota request --family p --type spot --desired-value 192 --region us-west-2

# --region is repeatable: submit the same increase in several regions at once.
# All target regions are validated up front — if any region's current quota is
# already >= the desired value, nothing is submitted.
aws-bootstrap quota request --type spot --desired-value 8 -r us-east-1 -r us-west-2

# Check request status (also repeatable across regions)
aws-bootstrap quota history --region us-west-2
aws-bootstrap quota history -r us-east-1 -r us-west-2

# Or use the AWS CLI directly:
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-3819A6DF \
  --desired-value 8 \
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

| Family | Type | Code | Description |
|--------|------|------|-------------|
| G/VT | Spot | `L-3819A6DF` | All G and VT Spot Instance Requests |
| G/VT | On-Demand | `L-DB2E81BA` | Running On-Demand G and VT instances |
| P | Spot | `L-7212CCBC` | All P Spot Instance Requests |
| P | On-Demand | `L-417A185B` | Running On-Demand P instances |
| DL | Spot | `L-85EED4F7` | All DL Spot Instance Requests |
| DL | On-Demand | `L-6E869C2A` | Running On-Demand DL instances |

Small increases (4-8 vCPUs) are typically auto-approved within minutes. You can also request increases via the [Service Quotas console](https://console.aws.amazon.com/servicequotas/home). While waiting, you can test the full launch/poll/SSH flow with a non-GPU instance type:

```bash
aws-bootstrap launch --instance-type t3.medium --ami-filter "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"
```

## Claude Code Plugin

A [Claude Code](https://docs.anthropic.com/en/docs/claude-code) plugin is included in the [`aws-bootstrap-skill/`](aws-bootstrap-skill/) directory, enabling LLM coding agents to autonomously provision and manage GPU instances.

### Install from GitHub

```bash
# Add the marketplace (registers this repo as a plugin source)
/plugin marketplace add promptromp/aws-bootstrap-g4dn

# Install the plugin
/plugin install aws-bootstrap-skill@promptromp-aws-bootstrap-g4dn
```

### Install locally (from repo checkout)

```bash
claude --plugin-dir ./aws-bootstrap-skill
```

See [`aws-bootstrap-skill/README.md`](aws-bootstrap-skill/README.md) for details.

## Additional Resources

| Topic | Link |
|-------|------|
| GPU instance pricing | [instances.vantage.sh](https://instances.vantage.sh/aws/ec2/g4dn.xlarge) |
| Spot instance quotas | [AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-limits.html) |
| Deep Learning AMIs | [AWS docs](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) |
| Nsight remote GPU profiling | [Guide](docs/nsight-remote-profiling.md) — Nsight Compute, Nsight Systems, and Nsight VSCE on EC2 |

Tutorials on setting up a CUDA environment on EC2 GPU instances:

- [Provision an EC2 GPU Host on AWS](https://www.dolthub.com/blog/2025-03-12-provision-an-ec2-gpu-host-on-aws/) (DoltHub, 2025)
- [AWS EC2 Setup for GPU/CUDA Programming](https://techfortalk.co.uk/2025/10/11/aws-ec2-setup-for-gpu-cuda-programming/) (TechForTalk, 2025)
