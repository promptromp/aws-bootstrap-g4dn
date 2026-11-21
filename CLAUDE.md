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
cp .envrc.example .envrc  # sample direnv config (venv activation + optional AWS_PROFILE)
direnv allow  # or manually: source .venv/bin/activate
```

`.envrc.example` is the tracked template; the copied `.envrc` is git-ignored.

## Project Structure

```
aws_bootstrap/
    __init__.py          # Package init
    cli.py               # Click CLI entry point (launch, status, terminate, list, quota, cluster commands)
    cluster.py           # Multi-node cluster composition: naming/rank helpers + launch fan-out (launch_cluster_nodes)
    config.py            # LaunchConfig dataclass with defaults
    ec2.py               # AMI lookup, security group, instance launch/find/terminate, polling, spot pricing, EBS volume ops, multi-region discovery/query, cluster placement-group + tag discovery
    gpu.py               # GPU architecture mapping (compute capability -> arch name) and GpuInfo dataclass
    output.py            # Output formatting: OutputFormat enum, emit(), echo/secho wrappers for structured output
    quota.py             # Service Quotas API: get/request GPU vCPU quotas (G/VT, P, DL families)
    retry.py             # Pure helpers: region precedence (resolve_regions), wait duration parsing, backoff schedule
    constants.py         # Centralized infra/tag literals (tag keys, resource types, ports, waiter configs, AMI owner IDs); re-exports config defaults
    ssh.py               # SSH key pair import, SSH readiness check, remote setup, ~/.ssh/config management, GPU queries, EBS mount
    resources/           # Artifacts SCP'd to remote instances (see ssh.SETUP_PAYLOAD)
        __init__.py
        gpu_benchmark.py       # GPU throughput benchmark (CNN + Transformer), copied to ~/gpu_benchmark.py on instance
        triton_vector_add.py   # Triton vector-add example, deployed to the instance
        gpu_smoke_test.ipynb   # Interactive Jupyter notebook for GPU verification, copied to ~/gpu_smoke_test.ipynb
        launch.json            # VSCode CUDA debug config template (deployed to ~/workspace/.vscode/launch.json)
        saxpy.cu               # Example CUDA SAXPY source (deployed to ~/workspace/saxpy.cu)
        tasks.json             # VSCode CUDA build tasks template (deployed to ~/workspace/.vscode/tasks.json)
        remote_setup.sh        # Uploaded & run on instance post-boot (GPU verify, Jupyter, etc.)
        torch_index.py         # Picks the PyTorch CUDA wheel index for the instance's toolkit (run by remote_setup.sh)
        requirements.txt       # Python dependencies installed on the remote instance
        cluster_canary.py      # Distributed DDP canary (all-reduce + SGD), SCP'd to cluster nodes for prepare/test
    tests/               # Unit tests (pytest)
        conftest.py            # Shared fixtures (runner, cli_session, quota/ami/instance-type rows)
        test_config.py
        test_cli.py
        test_cli_regions.py    # Region-aware quota/list CLI tests (fixtures + parametrize)
        test_cli_cluster.py    # cluster launch/status/prepare/test/run/terminate CLI tests
        test_cluster.py        # cluster.py composition helpers + run_distributed_job/canary orchestration
        test_ec2.py
        test_output.py
        test_gpu.py
        test_retry.py
        test_ssh_config.py
        test_ssh_exec.py       # run_on_host / scp_to_host / _ssh_opts SSH primitives
        test_ssh_keypair.py    # import_key_pair / generate_ssh_keypair
        test_constants.py      # constants <-> config re-export invariants
        test_ssh_gpu.py
        test_ebs.py
        test_ssh_ebs.py
        test_quota.py
        test_torch_index.py    # resources/torch_index.py CUDA-index discovery + SETUP_PAYLOAD wiring invariants
        test_impacted_rules.py # pytest-impacted rules cover every non-Python package file (see Coding Conventions)
        test_ec2_az.py         # AZ planning (plan_azs, spot prices, offerings, default subnets) + AZ-aware launch sweep
        test_cli_az.py         # --availability-zone / --cheapest-az / key-check wiring on launch + cluster launch
        test_ssh_keycheck.py   # ssh.check_key_usable (real ssh-keygen keys; agent/keychain/missing-tool paths)
docs/
    nsight-remote-profiling.md # Nsight Compute, Nsight Systems, and Nsight VSCE remote profiling guide
    capacity-and-retry.md      # Multi-region & --wait retry model, backoff design, recommended region lists
    multi-node-training.md     # End-to-end cluster training walkthrough (launch→prepare→test→run→terminate)
examples/                        # Runnable user-facing example scripts (not part of the package; ruff-linted, not type-checked)
    cluster/
        train_ddp.py             # Minimal multi-node DDP training script (long usage header)
        prepare_data.sh          # Idempotent per-node data-prep (S3→/data) for --data-script
        README.md
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

**Region resolution (most commands):** `--region` precedence is explicit flag(s) → `AWS_DEFAULT_REGION` / profile region → `us-west-2`. Implemented by `retry.resolve_regions` and the `resolve_region_list` / `resolve_single_region` helpers in `cli.py`. (Behavior change: the default is no longer hardcoded to `us-west-2` ignoring the profile.) The resolved/active region is echoed in `launch` and `terminate` output. **`status` and `cleanup` are exceptions:** they query *all enabled regions* (no `--region`), so they don't use single-region resolution — `cleanup` *must* be all-regions because the SSH config block records the instance ID but not its region, so staleness can only be judged against the global live set.

- **`launch`** — provisions an EC2 instance (spot by default); every non-dry-run result carries a `status` discriminator (`ready` | `no-public-ip` | `ssh-unavailable`) — the two *incomplete* outcomes leave a **running, billable** instance, so they now emit a structured record naming the `instance_id` and exit non-zero instead of returning silently with empty stdout and exit 0; `--region` is repeatable and tried in order (spot-first) on capacity shortfall; `--wait` retries with capped+jittered exponential backoff (30s→300s, ±20%) until `--wait-timeout` (default 30m; accepts `90s`/`30m`/`1h`/seconds) then hard-fails; quota / `SpotMaxPriceTooLow` are never waited on but warn and fall through to the next `--region`, hard-failing (aggregated, region-pinned hints) only when every region is blocked; without `--wait`, a single exhausted spot pass offers the on-demand fallback across all regions. Orchestrated by `ec2.launch_with_retry` (the sole launch path — the old single-region `ec2.launch_instance` primitive was unused and has been removed); adds SSH config alias (e.g. `aws-gpu1`) to `~/.ssh/config`; `--python-version` controls which Python `uv` installs in the remote venv; `--ssh-port` overrides the default SSH port (22) for security group ingress, connection checks, and SSH config; `--ebs-storage SIZE` creates and attaches a new gp3 EBS data volume (mounted at `/data`); `--ebs-volume-id ID` attaches an existing EBS volume (mutually exclusive with `--ebs-storage`) and **pins the instance to that volume's availability zone** (EBS is AZ-scoped — see EBS Data Volumes below); `--availability-zone/--az AZ` (repeatable; implies its region) and `--cheapest-az` (spot only) choose the AZ within each region — see *AZ selection* under Capacity below; `--skip-key-check` disables the pre-launch SSH key check (*Pre-launch SSH key check* below). The chosen AZ and its spot price are always reported (`spot_price_per_hour`), and `--dry-run` reports `availability_zones` per region
- **`status`** — lists all non-terminated instances (including `shutting-down`) with region, type, IP, SSH alias, EBS data volumes, pricing (spot price/hr or on-demand), uptime, and estimated cost for running spot instances. With no `--region`, queries **all enabled regions** (discovered via `describe_regions(AllRegions=False)`) in parallel and labels each instance with its region; pass one or more `--region`/`-r` values (repeatable) to restrict the query to those regions. Per-region query failures (e.g. an unauthorized region) emit a stderr warning and are skipped rather than aborting the command (and are surfaced as `regions_failed` in structured output). **Always sweeps the queried regions for orphaned EBS data volumes** (`find_orphan_ebs_volumes_in_regions`, parallel like the instance query) — detached tool-tagged volumes whose linked instance is gone bill silently and are invisible to the per-instance volume lookup, so status reports them (even with zero instances) with size, region, estimated monthly cost (`EBS_GP3_PRICE_PER_GB_MONTH` in `constants.py`), reattach hint, and a `cleanup --include-ebs` hint; structured output always carries `orphan_volumes` (own table in `table` mode — the instances table can't hold a second list). `--gpu` flag queries GPU info via SSH, reporting both CUDA toolkit version (from `nvcc`) and driver-supported max (from `nvidia-smi`); `--instructions` (default: on) prints connection commands (SSH, Jupyter tunnel, VSCode Remote SSH, GPU benchmark) for each running instance; suppress with `--no-instructions`
- **`terminate`** — terminates instances by ID or SSH alias (e.g. `aws-gpu1`, resolved via `~/.ssh/config`), or all aws-bootstrap instances in the region if no arguments given; removes SSH config aliases; deletes associated EBS data volumes by default; `--keep-ebs` preserves volumes and prints reattach commands. **Ownership guard:** explicitly-named targets are checked via `ec2.find_unowned_instances` for the `created-by` tag *before* anything is terminated, and the command hard-fails naming the offenders if any lack it (an explicit ID otherwise bypasses tag discovery entirely, so a typo or a wrong `--region` could destroy an unrelated instance); `--force` overrides. The no-argument path already selects by tag and skips the check
- **`cleanup`** — syncs `~/.ssh/config` with live instances **across all enabled regions** (`list_enabled_regions` + `find_tagged_instances_in_regions`, like `status`). Removes stale aliases (instances gone everywhere); **conservative guard:** if any region query fails, stale aliases are reported but *not* removed (could be alive in the unscanned region — surfaced as `regions_failed`). `--sync` additionally **adds** aliases for live instances missing one (cluster-tagged nodes get their deterministic `aws-<cluster-id>-<rank>` alias via `ClusterId`/`Rank` on `find_tagged_instances`; `--key-path`/`--ssh-user` for the stanza) and **repairs** aliases whose `HostName` drifted from the current public IP (`find_missing_ssh_hosts` / `find_drifted_ssh_hosts` in `ssh.py`). `--include-ebs` finds/deletes orphan EBS volumes across the scanned regions (via the same parallel `find_orphan_ebs_volumes_in_regions` helper `status` uses for read-only reporting); `--dry-run` previews (the diagnostic mode); `--yes` skips confirmation (required in structured modes). The single `--region` flag was removed (all-regions is mandatory for safe staleness). Never terminates instances.
- **`list instance-types`** — lists EC2 instance types matching a family prefix (default: `g4dn`), showing vCPUs, memory, **Quota Family** (the `gvt`/`p`/`dl` vCPU quota family each type draws from, via `instance_type_to_family`; `quota_family` in structured output, `null` for non-GPU), and GPU info; output tail prints copy-paste "Next steps" (region-pinned `quota show` / `quota request` for the family derived from `--prefix`, with a header stating the prefix→family mapping), suppressed for non-GPU families. **The filter is a true prefix match (`{prefix}*.*`)** — the previous `{prefix}.*` was an exact-family match that silently returned *nothing* for every family AWS names with a hyphenated suffix (`--prefix p6` found neither `p6-b200` nor `p6-b300`); as a side effect `g6` now also matches g6/g6e/g6f, and `g` matches every G family
- **`list amis`** — lists available AMIs matching a name pattern (default: `Deep Learning*Base OSS Nvidia Driver GPU AMI*` — every OS and architecture of the Deep Learning Base AMI, so `--arch arm64` alone finds the ARM64 builds), sorted newest-first; each AMI is labelled with its region (AMI IDs are region-specific). **Searches every CPU architecture by default**; `--arch x86_64|arm64` narrows it. (It used to hardcode `x86_64`, which made the ARM64 Deep Learning AMIs — the ones an arm64 type such as `g5g` needs — undiscoverable, while still printing an `Architecture` column.)
- **`quota show`** — displays current spot and on-demand vCPU quota values via AWS Service Quotas API; `--family` filters by instance family (gvt, p, dl; default: all)
- **`quota request`** — requests a quota increase; `--type` (spot/on-demand) and `--desired-value` are required; `--family` selects instance family (default: gvt); validates every target region's current value up front (aborting without submitting if any region's current ≥ desired); confirms in text mode, requires `--yes` in structured modes; structured output is `{"requests": [...]}` (one entry per region)
- **`quota history`** — shows history of quota increase requests; optional `--family`, `--type`, and `--status` filters

**Region handling for `list` / `quota` (show/request/history):** like other non-`status` commands, `--region`/`-r` follows the explicit → `AWS_DEFAULT_REGION`/profile → `us-west-2` precedence (via `resolve_region_list`), but is **repeatable**: pass several to query/submit across multiple regions. The active region(s) are always labelled in output (single region → `Region: <r>` line; multiple → a per-region block via the `_region_block_header` helper), and every structured record carries a `region` field. (`status` differs: no `--region` means *all enabled regions* — see its bullet.)

## Coding Conventions

- **Linting**: `ruff check` — line length 120, rules: E, F, UP, B, SIM, I, PLC
- **Formatting**: `ruff format` — double quotes, isort via ruff
- **Type checking**: `mypy` with `ignore_missing_imports = true`
- **Testing**: `pytest`. The pre-commit hook and pull-request CI run **only the tests a change can reach** ([pytest-impacted](https://github.com/promptromp/pytest-impacted), with the `[fast]` Rust parser). A push to main runs the whole suite.
- **All-in-one**: `pre-commit run --all` runs the full chain (ruff check, ruff format, mypy, impacted pytest)

After making changes, run:

```bash
pre-commit run --all
```

Or run tools individually:

```bash
uv run ruff check aws_bootstrap/
uv run ruff format aws_bootstrap/
uv run mypy aws_bootstrap/
uv run pytest                                              # every test
uv run pytest --impacted --impacted-git-mode=unstaged      # only what your uncommitted changes reach
```

### Impacted-test selection (pytest-impacted)

Configured in `pyproject.toml` `[tool.pytest.ini_options]`: `impacted_module = "aws_bootstrap"`, `impacted_tests_dir = "aws_bootstrap/tests"`.
- **Hook:** `--impacted-git-mode=unstaged` selects from git's state, not pre-commit's file list. A commit sees staged changes plus untracked files (pre-commit stashes unstaged edits first); `pre-commit run --all` also sees unstaged edits. A clean tree selects nothing and passes (the plugin turns pytest's exit 5 into 0), so run `uv run pytest` when you want the whole suite.
- **PR CI:** `--impacted-git-mode=branch --impacted-base-branch=origin/<base>`, diffed from the fork point (CI checks out with `fetch-depth: 0`).
- **Import analysis is automatic.** The plugin follows imports from `aws_bootstrap/` and the test files, so most changes need no configuration.
- **Files a test reads but doesn't import need a rule.** Examples: `test_cluster` reads `resources/cluster_canary.py`, and `test_torch_index` reads `remote_setup.sh` and checks the payload files. Add an `impacted_map` line (`GLOB=<test file>`), or an `impacted_invalidate_all` glob when any test may depend on it. A rule that is wrong must select more, never less.
- **The guard test catches a missing rule.** `tests/test_impacted_rules.py` fails when a tracked non-Python file under `aws_bootstrap/` (any depth) matches no rule, or a map target is missing. The map line `aws_bootstrap/**=…/test_impacted_rules.py` selects it for every package change. Without that line it would never run when it matters, since it imports no package code; it ignores that catch-all when checking coverage. Its limits:
  - It checks only that *some* rule matches, not that the rule names the test that reads the file.
  - It skips `.py` files, since import analysis covers them. A Python file a test reads as *data* (like `cluster_canary.py`) still needs its own map line.
- **Changes to dependencies or CI config run every test:** `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml` and `.github/workflows/**` invalidate all.

Use `uv add <package>` to add dependencies and `uv add --group dev <package>` for dev dependencies.

## Structured Output Architecture

The `--output` option uses a context-aware suppression pattern via `aws_bootstrap/output.py`:

- **`output.echo()` / `output.secho()`** — wrap `click.echo`/`click.secho`; silent in non-text modes. Used in `ec2.py` and `ssh.py` for progress messages.
- **`is_text(ctx)`** — checks if the current output format is text. Used in `cli.py` to guard text-only blocks.
- **`emit(data, headers=..., ctx=...)`** — dispatches structured data to JSON/YAML/table renderers. No-op in text mode.
- **CLI helper guards** — `step()`, `info()`, `val()`, `success()`, `warn()` in `cli.py` check `is_text()` and return early in structured modes.
- Each CLI command builds a result dict alongside existing logic, emits it via `emit()` for non-text formats, and falls through to text output for text mode.
- **Confirmation prompts** (`terminate`, `cleanup`, `cluster terminate`) require `--yes` in structured modes to avoid corrupting output.
- The spot-fallback `click.confirm()` in `ec2.py` auto-confirms in structured modes (via the `confirm_on_demand` callback default in `launch_with_retry`).

### ALWAYS verify a new command/field against all four `--output` formats

When adding or changing any CLI command (or the shape of its result), exercise it under **`text`, `json`, `yaml`, AND `table`** before considering it done — `json`/`yaml` passing does not imply `table` works. Concretely:

- Guard every human-readable line behind `is_text()` (use the `step/info/val/success/warn` helpers, and `output.echo/secho`); never `print`/`click.echo` unconditionally to **stdout**. The one exception: a warning that structured-mode users (often agents) must see may go to **stderr** unconditionally with `click.echo(..., err=True)`. It never corrupts the JSON/YAML on stdout. The SSH key check's warnings use this.
- Build one result dict and `emit()` it; surface errors via `CLIError` (goes to stderr, so stdout stays valid for parsers).
- **`table` mode needs `headers=`** when the result's primary payload is a list of dicts (e.g. `{"nodes": [...]}`): pass `emit(data, headers={key: "Label", ...})` so it unwraps and tabulates the inner list. Without `headers`, a dict-with-list renders the list as an unreadable Python `repr` in a single key-value cell (this exact bug shipped for the `cluster` commands — see PR #35). A flat dict (no list) or a list of plain strings is fine as key-value without headers.
- Any command with a confirmation prompt must require `--yes` in structured modes.
- Add a test that the command renders in `json`/`yaml`/`table` (parametrize on the format), not just `json`.

## Capacity, Multi-Region & Wait Retry

`ec2.launch_with_retry(config, prepare_region, ...)` is the launch orchestrator:

- **`prepare_region(region)`** — a `cli.py` callback returning a `RegionContext` (region-scoped boto3 client, AMI, security group); invoked at most once per region and cached across retries.
- **Spot-first per sweep** — each sweep calls `_run_instances` for every non-fatal region in `config.regions` order; first success returns a `RegionLaunch`. `_run_instances` raises `CapacityError` (`InsufficientInstanceCapacity`; retryable by next-region fallthrough *and* `--wait`) or `RegionFatalError` (quota via `_quota_error_message`, or `SpotMaxPriceTooLow`; `kind="quota"|"price"`).
- **`RegionFatalError` handling** — the region is recorded, dropped from later sweeps, and `on_region_fatal(region, kind, message)` fires so `cli.py` prints an immediate `WARNING` (region-pinned hint) before the next region is tried. Quota/price never trigger a `--wait` sleep. If every region is exhausted, `_aggregated_error` raises a single `CLIError` listing each region's reason + the full hint for each quota/price region.
- **`--wait`** — only sleeps while ≥1 region still has a retryable `CapacityError`; sleeps `retry.backoff_sleep_seconds(attempt)` (exponential, capped, jittered; clamped to the deadline) until `config.wait_timeout`, then hard-fails via `_aggregated_error`. Injectable `sleeper`/`clock`/`rng` keep it unit-testable. `on_wait(cycle, sleep_s, elapsed, retrying, skipped)` receives the regions actually re-swept vs. those dropped (quota/price), so the heartbeat reports the truth — it does NOT re-list quota-blocked regions as if still being attempted.
- **No `--wait`** — a fully-exhausted spot sweep triggers the on-demand fallback (`confirm_on_demand`) which sweeps **all** regions afresh (spot quota ≠ on-demand quota).
- **Region-aware errors** — `cli.prepare_region` prefixes setup failures with `[region]`; `validate_ebs_volume`, `resolve_ebs_placement_az`, and `quota._not_found_msg` name the region (EBS volumes and quotas are region-scoped).
- **Security-group ingress is re-checked on reuse — but only on groups we own.** `ensure_security_group` only *created* the SSH rule, so reusing an existing group with a different `--ssh-port` left the port closed and surfaced minutes later as an opaque "SSH did not become available". It now calls `_ensure_ssh_ingress` on the reuse path too (idempotent; `InvalidPermission.Duplicate` is success). **`_ensure_ssh_ingress` authorizes only when the group carries `created-by=<tag_value>`**: `--security-group` is user-supplied and the lookup matches on name + VPC alone, so a pre-existing restricted group can share the name — widening it to `0.0.0.0/0` would be a security change the user never asked for. An untagged group gets a warning with the exact `aws ec2 authorize-security-group-ingress` command and the launch continues (ingress may exist in a rule shape we don't model — a prefix list, or a security-group source). Never relax this to an unconditional authorize.
- **AMI architecture must match the instance type.** `ec2.instance_type_architecture(ec2_client, instance_type)` asks EC2 (`describe_instance_types` → `ProcessorInfo.SupportedArchitectures`) and `cli.prepare_region` threads the result into `get_latest_ami(..., architecture=...)` — for both `launch` and `cluster launch`. `get_latest_ami` previously hardcoded `x86_64`, so an arm64 GPU type (`g5g`, and the Grace-based P-series) got an x86_64 AMI and failed minutes later inside `run_instances`. Do **not** reintroduce a hardcoded architecture or a static ARM-family list — querying EC2 is what keeps this correct as AWS ships new families. Unknown types fall back to `x86_64` (preserving prior behaviour; the launch itself surfaces the authoritative error), and a filter that matches no AMI for the required architecture raises the arch-aware `_no_ami_message`, which names the architecture and prints a region-pinned `list amis --arch <arch> --region <r>`: with a custom filter it explains AWS's per-architecture naming and suggests dropping/fixing `--ami-filter`; with the default filter (which spans every architecture) it says the region doesn't publish the image and suggests another `--region`.
- **One AMI default for every architecture — `config.DEFAULT_AMI_FILTER`.** AWS names the Deep Learning Base AMI per architecture (`Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) <date>` for x86_64, `Deep Learning ARM64 Base OSS …` for arm64), so the default is `DLAMI_BASE_PATTERN` = `Deep Learning*Base OSS Nvidia Driver GPU AMI` + ` (Ubuntu 24.04)*`: the `*` spans the arch infix and the EC2 architecture filter picks the build. Verified (2026-11-21, us-west-2/us-east-1/ap-northeast-1, x86_64 + arm64) to resolve to the same image as AWS's public SSM pointer `/aws/service/deeplearning/ami/<arch>/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id`; it excludes Ubuntu Pro and the framework (PyTorch) DLAMIs. `list amis` defaults to `DLAMI_BASE_PATTERN*` (every OS and arch). Do **not** reintroduce per-architecture filter strings, and do not point users at the PyTorch DLAMIs — AWS does not patch CVEs in the 2.10–2.12 ones (DLAMI support policy).
- **AZ selection (`--availability-zone`, `--cheapest-az`).** `RegionContext.placement_azs` (a tuple; empty = EC2 picks) replaced the single `placement_az`: the `--ebs-volume-id` pin is `(vol_az,)`, a cluster's pin `(az,)`. `launch_with_retry`'s sweep tries each region's AZs in order. A capacity miss, a `SpotMaxPriceTooLow`, or an `AzUnsupportedError` (`Unsupported`, carrying AWS's message) moves to the next AZ *before* the next region, since all three can differ by AZ. **Quota is region-wide**: it skips the region's other AZs. The region counts as capacity-limited once every AZ missed and at least one on capacity; if every AZ failed on price/unsupported it is dropped (`kind="price"` if any AZ hit price, else `"unsupported"`). The aggregated error names only the AZs actually tried. **A single AZ is a hard pin** (an `--ebs-volume-id`, a cluster's AZ, a single `--az`) and keeps the pre-AZ-selection behaviour: `Unsupported` re-raises AWS's original `ClientError`, and a price failure drops the region. **No on-demand fallback when every failure is `unsupported`**: on-demand would hit the same AZs. That error's tail points at `list instance-types`. `on_attempt(region, market, attempt, az)` carries the AZ. The plan comes from `ec2.plan_azs(ec2, region, type, explicit, cheapest)`: candidates are the AZs offering the type (`describe_instance_type_offerings`) ∩ AZs with a default subnet (the tool launches into the default VPC; no filter if the region has none); explicit zones are confirmed with `describe_availability_zones` (unknown / not-opted-in → region-named `CLIError`) — `retry.region_of_az` (regex) only *routes* an AZ to its region, never validates it; `--cheapest-az` sorts by `spot_prices_by_az` (one paginated `describe_spot_price_history(StartTime=now)`, latest record per AZ; unknown prices last). Price semantics: ranking is **within** a region — region order still wins, and a capacity miss falls through to a pricier AZ (`--az X --wait` for cheap-or-nothing). Plans are computed for **every** `--region` up front, read-only and before anything is created or launched, so a bad AZ in region 2 can't surface after region 1 launched. They're cached across `--wait`, and dry-run reports them all. Explicit `--az` without `--cheapest-az` tolerates a denied `DescribeSpotPriceHistory` (prices are display-only); `--cheapest-az` needs it. The post-launch price is a fresh `spot_price_or_none`, falling back to the plan's (which may be an hour old after `--wait`). It is **best-effort and never raises** (it runs after RunInstances; losing the instance record to a throttled price call would break the incomplete-launch contract). Validation (`cli._validate_az_flags`): incompatible with `--ebs-volume-id`; `--cheapest-az` needs spot; an AZ outside the given `--region`s is an error. `cluster launch` applies them to a **new** cluster's first node only, planning before it creates the placement group or SG rule. Later nodes are pinned to that node's AZ, and a conflicting `--az` on an existing cluster is an error. `_emit_region_fatal` labels `quota` / `price` / `unsupported` separately; the `--wait` heartbeat names regions as blocked by quota/price/unsupported AZs.
- **Pre-launch SSH key check (`ssh.check_key_usable`).** Every ssh/scp runs with `BatchMode=yes`, so a passphrase key that nothing unlocks yields a billed instance setup can never reach. Where it runs: `cli.launch` (including `--dry-run`) calls it before any AWS call; `cluster launch` calls it after one read-only lookup (`find_cluster_instances`) and before creating anything. Both call it after the key exists or is generated; `--skip-key-check` bypasses it. The steps, in order:
  1. **Agent first.** If the agent ssh would ask holds the key's SHA256 fingerprint, pass, even with no private key file on disk (1Password-style). The fingerprint (`key_fingerprint`, from the `.pub` blob) matches `ssh-keygen -lf`; a non-OpenSSH `.pub` raises a `CLIError`. `_agent_sockets` follows ssh_config(5): an `identityagent` line from `ssh -G 203.0.113.1` **overrides** `$SSH_AUTH_SOCK` (`none` means no agent, the literal `SSH_AUTH_SOCK` means the env var, anything else is a path with `~`/`$VAR` expanded). Only without that line does `$SSH_AUTH_SOCK` count.
  2. **Private key missing:** error.
  3. **No passphrase:** `ssh-keygen -y -P "" -f <priv>` exits 0, so pass. `UNPROTECTED PRIVATE KEY FILE` gives a chmod error.
  4. **macOS `UseKeychain`:** if `platform.system() == "Darwin"` and `UseKeychain yes` is in `~/.ssh/config`, **warn only**. The Keychain may hold the passphrase, which can't be checked without side effects. Apple's `ssh -G` doesn't print it, and `Include`s aren't followed. Off macOS the setting can't help, so this case is an error.
  5. **Otherwise:** a `CLIError` with the fix (`ssh-add --apple-use-keychain` on macOS).
  - Missing ssh tools: warn and continue (fail-open).
  - Warnings go to stderr via `click.echo(err=True)` (the documented exception in *Structured Output Architecture*).
  - Tests: `conftest.key_check_stub` (autouse) stubs it for CLI tests that write only a `.pub`. Tests of the check itself use `@pytest.mark.real_key_check` or call it directly. Never run a live agent test that could inherit the user's `SSH_AUTH_SOCK`.
- **Robust key handling** (`ssh.import_key_pair`) — never mutates/deletes an existing AWS key pair (another machine may use it). It compares the existing pair's public key (`describe-key-pairs IncludePublicKey`) to the local `--key-path`; on mismatch it imports the local key under a deterministic derived name `<key-name>-<fp8>` (`fp8` = first 8 hex of the local key's SHA-256) and returns that. The effective name is threaded through `RegionContext.key_name` → `_run_instances`/`_build_launch_params` (NOT `config.key_name`), so each region launches with a key the user actually holds. A missing local key is auto-generated (`ssh.generate_ssh_keypair`, ed25519) in `cli.launch` instead of aborting.
- **No masked SSH/GPU errors** — `ssh._classify_ssh_failure` splits fatal auth/host-key failures from transient ones; `wait_for_ssh` fails fast on fatal (surfacing the real `ssh` stderr, not a generic "not ready") and reports the last error on timeout; `query_gpu_info` surfaces the failure reason. Reuse `_classify_ssh_failure` for any new SSH-retry logic rather than blind retry loops.
- **Pure policy in `retry.py`** — `resolve_regions`, `parse_duration`, `backoff_sleep_seconds` are side-effect-free and tested in `test_retry.py`. The CLI exposes `--wait-timeout` via the `_Duration` Click param type.
- `LaunchConfig` holds `regions: tuple[str, ...]` with a `region` property (`regions[0]`, falls back to `DEFAULT_REGION` if empty) for single-region callers; `wait` and `wait_timeout` fields drive the loop.
- **Text-mode color scheme** — `cli.py` helpers `_cmd()` (copy-paste commands → bold bright-cyan, used tool-wide), `_rtag()` / `_region_rule()` (per-region grouping → bold bright-blue), `_emit_region_fatal()` (bold-yellow verdict + cyan commands). All are `is_text()`-guarded, so structured output is never colorized/polluted. Reuse `_cmd()` for any new suggested-command output to stay consistent.

## Instance-Type & GPU Catalog Data

Prefer deriving catalog data from AWS (or from one in-repo source) over hand-kept
lists — every hand-kept list here has gone stale as AWS shipped new GPU families.
Last reconciled against the live AWS APIs on 2026-11-21 (`describe-instance-types`
across us-east-1/us-west-2/us-east-2/eu-west-1/ap-northeast-1 — 66 GPU types in 19
families, and `service-quotas get-service-quota` for all six codes):

- **`ec2._FAMILY_PREFIXES` / `instance_type_to_family`** — prefix-based (`g`/`vt`→gvt,
  `p`→p, `dl`→dl), so it needs **no** update for new families. Verified against the
  full 1389-type catalog: no non-accelerator family starts with those prefixes, and
  every current GPU family maps correctly (including hyphenated `p6-b200`/`p6-b300`,
  since the split is on `.`). `vt1` mapping to `gvt` despite having no `GpuInfo` is
  **correct** — VT1 genuinely draws the "G and VT" quota. `inf*`/`trn*` returning
  `None` is also correct: they have their own quotas, and this is a *GPU* family map.
- **`quota.QUOTA_FAMILIES`** — the six quota codes are verified against the live
  Service Quotas API. Do not "fix" them from memory or blog posts.
- **`quota.QUOTA_FAMILY_LABELS`** — *derived* from `_FAMILY_PREFIXES` (`"G and VT (g*, vt*)"`,
  `"P (p*)"`, `"DL (dl*)"`), so a label can't drift from the classification. It used to
  hand-list families and went stale twice (retired g3/p2/p3/dl1, then missing p6e-gb300).
  `list instance-types` shows the per-type mapping.
- **`ec2._gpu_summary`** — renders `GpuInfo` straight from the API. Fractional-GPU types
  (g6f, gr6f) report `Count: 0` plus `GpuPartitionSize`, shown as e.g. `1/8 L4`.
  `trn3` also reports `GpuInfo` (Annapurna Trainium3); it has no quota family, correctly.
- **`gpu._GPU_ARCHITECTURES`** — the one remaining hand-kept display table (compute
  capability → architecture name). No `nvidia-smi --query-gpu` field reports the
  architecture, so it stays; it falls back to `Unknown (X.Y)`, so a missing entry degrades
  gracefully — only add one you have actually confirmed (e.g. sm_103 = GB300 "Blackwell
  Ultra", CUDA 12.9+). Covers every current GPU family through Blackwell (sm_120, g7/g7e).

## CUDA-Aware PyTorch Installation

`remote_setup.sh` detects the CUDA toolkit version on the instance (via `nvcc`, falling back to `nvidia-smi`) and installs PyTorch from the matching CUDA wheel index (`https://download.pytorch.org/whl/cu{TAG}`). This ensures `torch.version.cuda` matches the system's CUDA toolkit, which is required for compiling custom CUDA extensions with `nvcc`.

**The tag is discovered, not listed.** `resources/torch_index.py` (stdlib-only; uploaded via `SETUP_PAYLOAD` and run with the venv's python) reads the `cuNNN/` directories PyTorch publishes at https://download.pytorch.org/whl/, keeps those of the toolkit's CUDA **major** that are ≤ its major.minor (numeric compare — `13.5.1` → `(13, 5)`; a 13.x toolkit never gets a 12.x index, which would break `nvcc` extension builds), and runs `uv pip compile` against each, newest first, accepting the first where **every** package (`torch`, `torchvision`) resolves to a `+cuNNN` build. **Only a definite no-build moves on** (uv exit 1 → `NoBuild`, or a resolution landing on a CPU wheel); uv exit 2 / timeout / missing uv → `ResolverUnavailable` stops the search and falls back to PyPI — walking older indexes on a network failure would stall setup and silently downgrade. Prints the tag or nothing (→ the existing PyPI fallback); each rejected tag is logged to stderr. The resolution check matters: an index can exist before it serves every package for this python/arch, and the indexes mirror stray CPU-only wheels that would otherwise "resolve". Wheel compatibility is left to `uv` — don't hand-parse wheel filenames (torchvision ships `py3-none` wheels, not `cpXY`). It replaced a hardcoded `KNOWN_CUDA_TAGS` list that had silently pinned every launch to `cu130` while the DLAMI moved to CUDA 13.5 (as of 2026-11-21: x86 Base AMI default CUDA 13.5 → `cu134`; ARM64 Base AMI 13.2 → `cu132`). The index fetch (`FETCH_TIMEOUT` 30s) and each resolve (`RESOLVE_TIMEOUT` 120s; a healthy one takes ~1s) are bounded, and the same-major rule caps attempts at a handful (`remote_setup.sh` itself runs unbounded). `tests/test_torch_index.py` covers it with fakes, plus the invariant that every `/tmp/<file>` `remote_setup.sh` reads is in `SETUP_PAYLOAD`. To check it off-instance: `python aws_bootstrap/resources/torch_index.py 13.5 --python $(uv python find 3.12) --python-platform x86_64-manylinux_2_28`.

`torch` and `torchvision` are **not** in `resources/requirements.txt` — they are installed separately by the CUDA detection logic in `remote_setup.sh`. All other Python dependencies remain in `requirements.txt`.

## Remote Setup Details

`remote_setup.sh` also:
- Creates `~/venv` and appends `source ~/venv/bin/activate` to `~/.bashrc` so the venv is auto-activated on SSH login. When `--python-version` is passed to `launch`, the CLI sets `PYTHON_VERSION` as an inline env var on the SSH command; `remote_setup.sh` reads it to run `uv python install` and `uv venv --python` with the requested version
- Adds NVIDIA Nsight Systems (`nsys`) to PATH if installed under `/opt/nvidia/nsight-systems/` (pre-installed on Deep Learning AMIs but not on PATH by default). Fixes directory permissions, finds the latest version, and prepends its `bin/` to PATH in `~/.bashrc`
- Runs a quick CUDA smoke test (`torch.cuda.is_available()` + GPU matmul) after PyTorch installation to verify the GPU stack; prints a WARNING on failure but does not abort
- Writes the detected CUDA version to `~/.aws-bootstrap-cuda` (a sentinel: the literal `none` when no CUDA is found). After a successful setup the CLI reads it back over SSH via `ssh.query_cuda_version` and shows `CUDA Version: X.Y` in the final "Instance ready!" summary (and as `cuda_version` in structured output). Degrades silently: `--no-setup` never queries it; a non-`\d+\.\d+` value (e.g. `none` on a non-DL/smoke-test box) or any SSH failure → line omitted, never an error
- Copies `gpu_benchmark.py` to `~/gpu_benchmark.py` and `gpu_smoke_test.ipynb` to `~/gpu_smoke_test.ipynb`. Help text that runs the benchmark over one-shot SSH uses `~/venv/bin/python` (a non-interactive `ssh host 'cmd'` does NOT source `~/.bashrc`/activate the venv, and Ubuntu 24.04 has no unversioned `python`)
- Sets up `~/workspace/.vscode/` with `launch.json` and `tasks.json` for CUDA debugging. Detects `cuda-gdb` path and GPU SM architecture (via `nvidia-smi --query-gpu=compute_cap`) at deploy time, replacing `__CUDA_GDB_PATH__` and `__GPU_ARCH__` placeholders in the template files via `sed`

## GPU Benchmark

`resources/gpu_benchmark.py` is uploaded to `~/gpu_benchmark.py` on the remote instance during setup. It benchmarks GPU throughput with two modes: CNN on MNIST and a GPT-style Transformer on synthetic data. It reports samples/sec, batch times, and peak GPU memory. Supports `--precision` (fp32/fp16/bf16/tf32), `--diagnose` for CUDA smoke tests, and separate `--transformer-batch-size` (default 32, T4-safe). Dependencies (`torch`, `torchvision`, `tqdm`) are already installed by the setup script.

## EBS Data Volumes

The `--ebs-storage` and `--ebs-volume-id` options on `launch` create or attach persistent gp3 EBS volumes mounted at `/data`. The implementation spans three modules:

- **`ec2.py`** — Volume lifecycle: `create_ebs_volume`, `validate_ebs_volume`, `attach_ebs_volume`, `delete_ebs_volume`, `find_ebs_volumes_for_instance`, plus `resolve_ebs_placement_az` (AZ pinning, see below). `EBS_DEVICE_NAME` (`/dev/sdf`) and `EBS_MOUNT_POINT` (`/data`) are defined in `constants.py` (`EBS_DEVICE_NAME` is also imported into `ec2.py` as an attach default).
- **`ssh.py`** — `mount_ebs_volume()` SSHs to the instance and runs a shell script that detects the device, optionally formats it, mounts it, and adds an fstab entry.
- **`cli.py`** — Orchestrates the flow: resolve AZ (existing volume) → create/validate → attach → wait for SSH → mount. Mount failures are non-fatal (warn and continue).

### Availability-zone pinning (`--ebs-volume-id`)

EBS volumes are **AZ-scoped** — an instance can only attach a volume in its own availability zone. Without pinning, a launch lands in a random AZ within the region and, if it differs from the volume's AZ, `attach_ebs_volume` fails *after* the instance is already running. To prevent this:

- `ec2.resolve_ebs_placement_az(ec2_client, volume_id, region)` looks up the volume's AZ once (read-only `describe_volumes`), raising a region-named `CLIError` (shared `_ebs_volume_not_found_message` helper, also used by `validate_ebs_volume`) if the volume isn't in that region.
- `cli.prepare_region` calls it when `config.ebs_volume_id` is set and stores the AZ on `RegionContext.placement_az` (resolved once per region, cached across `--wait` retries — never re-fetched in the hot loop).
- `launch_with_retry`'s sweep threads `ctx.placement_az` into `_run_instances` → `_build_launch_params`, which sets `Placement.AvailabilityZone` on the `run_instances` request.
- The post-launch `validate_ebs_volume` AZ check remains as defense-in-depth (it now passes because the instance is pinned correctly).
- **Tradeoff:** pinning constrains spot capacity to that single AZ rather than the whole region, so `InsufficientInstanceCapacity` is more likely; `--wait` still helps as that AZ's capacity frees up. A `--ebs-volume-id` launch is effectively single-region (the volume lives in exactly one region/AZ).

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

## Multi-Node Training Clusters (`cluster` group)

The `cluster` command group (`cluster.py` + the `cluster` Click group in `cli.py`) launches and manages multiple GPU instances as a coordinated cluster for multi-node distributed training (`torchrun`). Ships the full MVP: `cluster launch` / `status` / `prepare` / `test` / `run` / `terminate`.

- **`cluster.py`** — composition logic only: pure helpers (`placement_group_name`, `node_alias`, `nodes_to_add`) and the launch fan-out (`launch_cluster_nodes`, which calls an injectable `launch_fn` per node and assigns sequential ranks). It imports `ec2` (one-directional: `ec2.py` does **not** import `cluster.py`, which is why the placement-group helpers take an already-derived name rather than a `cluster_id`).
- **Identity = tags, not state.** A cluster is an arbitrary `--cluster-id` stored as the `TAG_CLUSTER_ID` (`aws-bootstrap-cluster`) tag; each node also carries `TAG_CLUSTER_RANK` (`aws-bootstrap-cluster-rank`). `find_cluster_instances` / `list_clusters` (in `ec2.py`) discover and group by these tags. No local cluster state file.
- **Single AZ + cluster placement group.** `ensure_cluster_placement_group(ec2, name, tag_value)` creates/reuses a `cluster`-strategy group; `cli.cluster_launch` captures the first node's AZ and pins the rest via `RegionContext.placement_az` + `RegionContext.placement_group` (both threaded through `_build_launch_params`/`_run_instances`/`launch_with_retry` alongside the existing `placement_az`).
- **Intra-cluster SG rule.** `ensure_cluster_security_group_rule(ec2, sg_id)` adds an idempotent self-referencing (`-1`/all-protocol) ingress rule so members can reach each other on NCCL/rendezvous ports.
- **SSH aliases.** Each node gets a deterministic `aws-<cluster-id>-<rank>` alias via `add_ssh_host(..., alias=...)` (the explicit-alias parameter added for clusters).
- **`RDZV_PORT`** (`29400`, in `constants.py`) is the reserved torchrun c10d rendezvous port for later phases.

`cluster launch` reuses the single-node launch machinery (`launch_with_retry`, `ensure_security_group`, `import_key_pair`, `get_latest_ami`, `wait_instance_ready`, `remote_setup`) per node; the AMI/key are looked up once and reused across nodes. `cluster launch` is **incremental**: re-running with a higher `--nodes` adds nodes to reach the target. EFA is out of scope (P-series only; `g4dn`/`g5` use plain VPC networking).

### Prepare / test (distributed execution core)

- **c10d rendezvous, identical command per node.** `cluster.build_torchrun_command(...)` builds one `torchrun --rdzv-backend=c10d --rdzv-endpoint=<rank0_private_ip>:RDZV_PORT --rdzv-id=<cluster-id> ...` command that is run on **every** node; the rendezvous (hosted by rank 0) assigns global ranks, so no per-node `--node-rank` is needed. `cluster.master_addr(nodes)` returns rank 0's private IP.
- **Parallel start.** `cluster.run_on_all_nodes(nodes, command_for, run_fn=...)` runs the command on all nodes concurrently (a `ThreadPoolExecutor`) — they must start together to rendezvous — returning a rank-ordered `list[NodeResult]`. `run_fn`/`scp_fn` are **injected** (production wraps `ssh.run_on_host` / `ssh.scp_to_host`), which keeps the orchestration unit-testable with fakes.
- **Generic SSH primitives** live in `ssh.py`: `run_on_host(host, user, key, command, *, port, timeout) -> (rc, stdout, stderr)` and `scp_to_host(...) -> bool`.
- **Canary.** `resources/cluster_canary.py` is a tiny DDP job (process-group init + all-reduce correctness check + a few SGD steps; exits non-zero on failure). `cluster.run_canary(...)` SCPs it to every node and runs the torchrun command in parallel (activating `~/venv` first, since non-interactive ssh doesn't source `~/.bashrc`). `cluster test` runs it standalone; `cluster prepare` runs it after verification.
- **`cluster prepare`** queries each node's GPU/CUDA via `query_gpu_info`, fails on `cluster.detect_version_skew` (inconsistent CUDA across nodes), writes the `AWSB_*` env contract to `~/.aws-bootstrap-cluster` on each node as a **sourceable** file (`cluster.render_node_config` → `export K=<shlex.quote(v)>` lines, so the newline-joined `AWSB_NODE_IPS` doesn't leak a bare line), then auto-runs the canary (`--no-canary` skips it).
- **`RDZV_PORT`** (`29400`) is the c10d rendezvous port; the SG self-rule (Phase 1) makes it reachable between nodes.

### Run (distributed training jobs)

- **`cluster.run_distributed_job(...)`** is the general runner; `run_canary` now delegates to it (DRY). Steps: SCP the script to all nodes → (optional) SCP+run a `--data-script` on every node in parallel as a **barrier** (training won't start until all preps succeed) → run the training command on all nodes in parallel.
- **`_job_command_for(node, ...)`** picks the per-node command: a `.py` script → `build_torchrun_command` (identical on every node, c10d); a `.sh` script → the **escape hatch** (`node_env` `AWSB_*` exports + `bash script.sh`, so the user can drive any launcher). Both prepend `source ~/venv/bin/activate`.
- **`cli.cluster_run`** resolves nodes, calls `run_distributed_job`, writes per-node logs to `<--log-dir>/<cluster-id>/rank<N>.log`, echoes stdout from **every** node that produced output (under c10d rendezvous torch's global rank 0 is NOT necessarily node rank 0, so we can't single out one node), and exits non-zero if any node failed. `context_settings={"ignore_unknown_options": True}` + a trailing `script_args` (`nargs=-1`) lets training args pass through after `--`.
- **`--wait`/`--wait-timeout`** on `cluster launch` thread into `LaunchConfig`, so `launch_with_retry`'s backoff loop retries each node on spot capacity shortfall and hard-fails on timeout (no on-demand prompt) — important for non-interactive/cluster use.
- **`cluster terminate`** terminates the nodes, removes SSH aliases, then **waits** (`instance_terminated` waiter) for full termination before deleting the placement group — a placement group can't be deleted while instances are still `shutting-down` (`delete_cluster_placement_group` returns `False` on `InvalidPlacementGroup.InUse`, and the command warns rather than crashing).
- **Execution commands require every node `running`.** `find_cluster_instances` reports all non-terminated nodes (so `status` can show a stopped one), but `cluster prepare`/`test`/`run` go through `cli._running_cluster_nodes`, which hard-fails with an actionable message if any node is `stopped`/`shutting-down`. A distributed job needs every rank at the rendezvous, and a spot-reclaimed rank 0 is still what `master_addr` picks — the whole job would otherwise hang against an IP that is going away.
- **Deferred (not yet):** `--detach`/live log streaming/`cluster logs`/`--poll-interval` (the synchronous run with per-node log files is the MVP); auto-detecting `--nproc-per-node` from `prepare`'s written config (defaults to 1).

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

### SSH subprocess invariants

All ssh/scp option construction goes through `ssh._ssh_opts` (identity, no host-key
prompts, `BatchMode=yes`, `ConnectTimeout`) and `ssh._port_opts` (`-p` for ssh, `-P` for
scp). Two rules when touching them:

- **`BatchMode=yes` is not optional.** Without it a key-mismatched or unreachable host
  makes a non-interactive ssh/scp block on a password prompt with the CLI's stdin.
- **`ConnectTimeout` lives only in `_ssh_opts`.** ssh honors the *first* `-o` occurrence,
  so a caller appending its own after these options would be silently ignored — pass
  `_ssh_opts(key, connect_timeout=N)` instead.

Every `subprocess.run` that captures output is bounded by a `timeout=` (`SCP_TIMEOUT` for
transfers); only the two deliberately-streaming calls (`run_remote_setup`'s script
execution, `mount_ebs_volume`) run unbounded, and both are interactive by design.

`ssh.SETUP_PAYLOAD` lists the files uploaded to `/tmp` before `remote_setup.sh` runs — add
new resources there rather than writing another copy-pasted SCP block.

### Agentic coding artifacts

Do **not** commit superpowers or other agentic-coding plan artifacts (design specs, implementation plans, scratch notes under `docs/superpowers/`, etc.) to the repo unless there's a specific reason or the user explicitly asks. `docs/superpowers/` is git-ignored; keep these local. Real, user-facing project documentation belongs in `README.md`, `CLAUDE.md`, or `docs/`.
