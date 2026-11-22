# aws-bootstrap Command Reference

Complete option documentation and JSON output schemas for all commands.

## Global Options

These options go **before** the command name:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output` / `-o` | `text\|json\|yaml\|table` | `text` | Output format |
| `--version` | flag | | Show version and exit |
| `--help` | flag | | Show help and exit |

Per-command options `--region` and `--profile` are available on all commands:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--region` / `-r` | string | `AWS_DEFAULT_REGION`/profile region, then `us-west-2` | AWS region. Precedence: explicit flag → env/profile region → `us-west-2`. **Repeatable** on `launch` (tried in order), `quota show`/`request`/`history`, and `list instance-types`/`amis` (queried/submitted per region; each structured record carries a `region`). On `status` it is also repeatable and, when omitted, defaults to *all enabled regions* (see below) instead of a single region. |
| `--profile` | string | `AWS_PROFILE` env | AWS profile override |

The active region(s) are shown in `launch`, `terminate`, `quota`, and `list` output (single region → a `Region:` line; multiple → a per-region block). `status` and `cleanup` always query **all enabled regions** (they report `regions_queried`/`regions_failed` instead of a single region label).

---

## `aws-bootstrap launch`

Provision a GPU-accelerated EC2 instance.

```
aws-bootstrap launch [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--instance-type` | string | `g4dn.xlarge` | EC2 instance type |
| `--ami-filter` | string | auto-detected | AMI name pattern filter |
| `--spot` / `--on-demand` | flag | `--spot` | Pricing model |
| `--region` | string (repeatable) | env/profile, then `us-west-2` | Repeat to try regions in order, spot-first, on capacity shortfall |
| `--wait` | flag | false | On insufficient spot capacity, retry with bounded exponential backoff until `--wait-timeout`, then hard-fail |
| `--wait-timeout` | duration | `30m` | Max wait when `--wait` set; accepts `90s`, `30m`, `1h`, or bare seconds |
| `--key-path` | path | `~/.ssh/id_ed25519.pub` | SSH public key path |
| `--key-name` | string | `aws-bootstrap-key` | AWS key pair name |
| `--security-group` | string | `aws-bootstrap-ssh` | Security group name |
| `--volume-size` | int | `100` | Root EBS volume size (GB, gp3) |
| `--no-setup` | flag | false | Skip remote setup script |
| `--dry-run` | flag | false | Validate without launching |
| `--python-version` | string | none | Python version for remote venv (e.g. 3.13) |
| `--ssh-port` | int | `22` | SSH port for security group and connection |
| `--ebs-storage` | int | none | Create new EBS data volume (GB, gp3, at /data) |
| `--ebs-volume-id` | string | none | Attach existing EBS volume (at /data); the volume decides the region and AZ |
| `--iam-instance-profile` | string | none | Attach an existing IAM instance profile (name or ARN) so the instance gets AWS credentials (e.g. S3); never creates one |
| `--availability-zone` / `--az` | string (repeatable) | none | Only these AZs, tried in order; an AZ implies its region |
| `--cheapest-az` | flag | false | Spot only: try each region's AZs cheapest-first by current spot price |
| `--skip-key-check` | flag | false | Skip the pre-launch check that the SSH key works without a prompt |

`--ebs-storage` and `--ebs-volume-id` are mutually exclusive. `--availability-zone` / `--cheapest-az` can't be combined with `--ebs-volume-id` (the volume pins the AZ), and `--cheapest-az` needs spot.

**AZ selection:** spot prices can differ several-fold between AZs of one region. With `--cheapest-az`, each region's AZs that offer the type are ranked by current spot price. A capacity miss (or a per-AZ spot price or support failure) moves to the next AZ in the same region before the next region. Ranking is **within** a region: region order still wins, and the fallthrough AZ may be pricier. With `--ebs-storage`, the new volume lives in whichever AZ the instance lands in. For "the cheap AZ or nothing", use `--availability-zone <az> --wait`. The chosen AZ and its spot price are always reported (`availability_zone`, `spot_price_per_hour`). Check the plan first with `--dry-run`, which reports `availability_zones` per region.

**Pre-launch key check:** aws-bootstrap connects with `BatchMode=yes` (no prompts). Before any AWS call, including in `--dry-run`, `launch` checks that the agent ssh would use holds the key, including via `IdentityAgent` such as 1Password, or that the key has no passphrase. Otherwise it exits 1 *before* creating a billed instance and prints the fix (`ssh-add --apple-use-keychain <key>`). On macOS with `UseKeychain yes` in `~/.ssh/config`, it only warns on stderr, because the Keychain may hold the passphrase. On this error, don't retry blindly: load the key into the agent (the user may need to do this in their own terminal), or pass `--skip-key-check` only if you know the passphrase is supplied another way.

`--ebs-volume-id` finds the volume before anything is created (in the `--region`s given, or the default region, then every enabled region) and launches in its region, pinned to its availability zone (EBS volumes are AZ-scoped) — no `--region` needed. Spot capacity is limited to that one AZ, so combine with `--wait` if it's short. Errors, all before provisioning anything: the volume is outside an explicit `--region` (the message names its real region — drop `--region` or use that one); it isn't `available` (e.g. still attached to an instance that is shutting down — the message names it; wait and retry); a malformed ID; not found in any region (unqueryable regions are named).

**`--iam-instance-profile`:** checked before anything is created. "not found" stops the launch (exit 1): list profiles with the printed `aws iam list-instance-profiles` command, don't guess names. A stderr warning that it "could not verify" (no `iam:GetInstanceProfile`) or that it has no role does not stop the launch. If EC2 rejects the profile at launch (e.g. created seconds ago, or no `iam:PassRole`), the launch stops at once with that error — no other region, no on-demand fallback, no `--wait` retry. For an already-running instance use `aws ec2 associate-iam-instance-profile` (there is no aws-bootstrap command for it).

**`--wait` + multiple `--region`:** a region sweep is the inner loop, backoff is the outer loop. Each cycle tries spot in every `--region` in order with no delay between regions; only when *all* regions miss does it sleep (capped+jittered exponential backoff, escalating per sweep) and sweep again, until `--wait-timeout` total wall-clock, then hard-fail. So `--wait --region A --region B` = "try A then B instantly; if both dry, back off and retry both" — not "wait on A then try B." Region order wins every tie. Without `--wait`, exactly one sweep then on-demand fallback.

### JSON Output

**Normal launch:**
```json
{
  "status": "ready",
  "instance_id": "i-0abc123def456",
  "public_ip": "54.200.1.2",
  "instance_type": "g4dn.xlarge",
  "availability_zone": "us-west-2a",
  "ami_id": "ami-0abc123",
  "pricing": "spot",
  "region": "us-west-2",
  "regions_tried": ["us-west-2", "us-east-1"],
  "ssh_alias": "aws-gpu1",
  "spot_price_per_hour": 0.9911,
  "on_demand_price_per_hour": 0.526,
  "iam_instance_profile": "my-ec2-s3-profile",
  "cuda_version": "13.2",
  "ebs_volume": {
    "volume_id": "vol-0abc123",
    "mount_point": "/data",
    "size_gb": 96
  }
}
```

`status` is the outcome discriminator, present on every non-dry-run `launch` result: `"ready"` on success, `"no-public-ip"` or `"ssh-unavailable"` on an **incomplete launch**. The `ebs_volume` field is only present when `--ebs-storage` or `--ebs-volume-id` is used. `cuda_version` is present only when remote setup ran and a CUDA version was detected (omitted with `--no-setup` or on non-CUDA/smoke-test instances). `region` is the region the instance actually launched in; `regions_tried` lists all regions attempted in order. `spot_price_per_hour` is the current spot price in the chosen `availability_zone` (spot launches only; omitted if the lookup fails, which never fails the launch). `on_demand_price_per_hour` is the region's on-demand price for the type, for comparison: present for either market, omitted if unknown. `iam_instance_profile` (the profile's name) is present only with `--iam-instance-profile`, also on an incomplete launch.

**Incomplete launch** (exit code 1 — the instance **exists and is billing**, but is unusable):
```json
{
  "status": "ssh-unavailable",
  "instance_id": "i-0abc123def456",
  "public_ip": "54.200.1.2",
  "instance_type": "g4dn.xlarge",
  "availability_zone": "us-west-2a",
  "region": "us-west-2",
  "pricing": "spot",
  "error": "SSH did not become available within the timeout."
}
```

`status` is `"ssh-unavailable"` (instance up, SSH never answered) or `"no-public-ip"` (no
public IP assigned — check VPC settings). Unlike other failures, stdout is **not** empty:
the record carries the `instance_id` so the caller can retry the connection or terminate
the instance rather than leaking it. Branch on `status`, not on the exit code alone.

**Dry run:**
```json
{
  "dry_run": true,
  "instance_type": "g4dn.xlarge",
  "ami_id": "ami-0abc123",
  "ami_name": "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) ...",
  "pricing": "spot",
  "key_name": "aws-bootstrap-key",
  "security_group": "sg-0abc123",
  "volume_size_gb": 100,
  "regions": ["us-west-2"],
  "wait": false,
  "wait_timeout_seconds": 1800,
  "availability_zones": {
    "us-west-2": [
      {"zone": "us-west-2b", "spot_price": 0.9911, "placement_score": 9},
      {"zone": "us-west-2a", "spot_price": 2.2122, "placement_score": 3}
    ]
  },
  "on_demand_price_per_hour": {"us-west-2": 0.526},
  "spot_pools": {
    "us-west-2": [
      {"zone": "us-west-2a", "spot_price": 2.2122, "placement_score": 3},
      {"zone": "us-west-2b", "spot_price": 0.9911, "placement_score": 9}
    ]
  },
  "quota_fit": {
    "us-west-2": {
      "family": "gvt",
      "spot": {"quota": 8, "in_use": 4, "needed": 4, "fits": true},
      "on-demand": {"quota": 8, "in_use": 8, "needed": 4, "fits": false}
    }
  }
}
```

**Choosing a pool from a dry run.** These are display only and never change what launches:
- `on_demand_price_per_hour` per region;
- `spot_pools` per region (spot only): every usable AZ with its spot price and **placement score** (in price order with `--cheapest-az`, else by zone). The score is EC2's 1–10 estimate that a spot request succeeds there; a low-score cheapest AZ often has no capacity. Scores are given only for P, G, Trn and Inf types (elsewhere `placement_score` is `null`): AWS caps single-type scores low for other families.
- `quota_fit` per region: an estimate of whether one more instance fits each market's vCPU quota, given running instances. `fits: false` means the launch would likely be refused in that market: `aws-bootstrap quota show` confirms the limit; in-use is estimated from running instances (open spot requests and unused capacity reservations aren't counted). Text mode prints the `quota request` that would make room.

Prefer a high-score AZ with `--az <zone>`; `--cheapest-az` ranks by price only. Any of these is `null` (or `unknown` in text) when its lookup isn't permitted (`pricing:GetProducts`, `ec2:GetSpotPlacementScores`, `servicequotas:GetServiceQuota`). AZ names are per account.

With `--ebs-volume-id`, the dry run also carries `ebs_volume_id` and `ebs_volume_availability_zone`, and `regions` is the volume's region; with `--iam-instance-profile`, `iam_instance_profile`.

`availability_zones` is `null` unless `--availability-zone` or `--cheapest-az` is given. It maps each `--region` to the AZs that would be tried, in order, or `null` where EC2 chooses. In `-o table` it is a single string.

On capacity timeout (`--wait` exhausted) or all regions exhausted, `launch` exits non-zero with an aggregated `CLIError` (per-region reasons + region-pinned quota hints). Quota / `SpotMaxPriceTooLow` are never *waited* on, but in multi-region mode the launcher warns and moves on to the next `--region`; it only fails hard once every region is blocked.

---

## `aws-bootstrap status`

Show running instances created by aws-bootstrap.

```
aws-bootstrap status [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--region` / `-r` | string (repeatable) | all enabled regions | Region(s) to query. Repeat for multiple. If omitted, queries every region enabled for the account. |
| `--gpu` | flag | false | Query GPU info (CUDA, driver) via SSH |
| `--instructions` / `--no-instructions` / `-I` | flag | true | Show connection commands for running instances |

> Note: unlike other commands, `status` `--region` is **repeatable** and defaults to *all enabled regions* (not `us-west-2`). Each instance is labelled with its `region`.

### JSON Output

```json
{
  "instances": [
    {
      "instance_id": "i-0abc123def456",
      "region": "us-west-2",
      "state": "running",
      "instance_type": "g4dn.xlarge",
      "public_ip": "54.200.1.2",
      "ssh_alias": "aws-gpu1",
      "lifecycle": "spot",
      "availability_zone": "us-west-2a",
      "iam_instance_profile": "my-ec2-s3-profile",
      "launch_time": "2025-01-15T10:30:00+00:00",
      "spot_price_per_hour": 0.1578,
      "uptime_seconds": 3600,
      "estimated_cost": 0.1578,
      "gpu": {
        "name": "Tesla T4",
        "architecture": "Turing",
        "cuda_toolkit": "12.8",
        "cuda_driver_max": "13.0",
        "driver": "570.86.15"
      },
      "ebs_volumes": [
        {
          "volume_id": "vol-0abc123",
          "size_gb": 96,
          "mount_point": "/data",
          "state": "in-use"
        }
      ]
    }
  ],
  "regions_queried": ["us-east-1", "us-west-2"],
  "regions_failed": [
    {"region": "ap-south-1", "error": "AuthFailure: ..."}
  ],
  "orphan_volumes": [
    {
      "volume_id": "vol-0def456",
      "region": "us-west-2",
      "size_gb": 150,
      "monthly_cost_usd": 12.0,
      "last_instance_id": "i-0dead123"
    }
  ]
}
```

Top-level `regions_queried` lists every region the query was attempted against (including any that subsequently failed). `regions_failed` is present only when one or more regions could not be queried (e.g. unauthorized); those regions appear in both lists. Per-instance fields `gpu`, `ebs_volumes`, `spot_price_per_hour` (spot) / `on_demand_price_per_hour` (on-demand, from the Price List API), `uptime_seconds`, and `estimated_cost` (running instances whose price is known) are conditional. `iam_instance_profile` is always present: the attached profile's name, or `null`.

`orphan_volumes` is **always present** (empty list when none): aws-bootstrap data volumes that are detached and whose linked instance no longer exists — they still bill (`monthly_cost_usd` is an estimate at the common gp3 rate) until reattached (`launch --ebs-volume-id <id> --region <region>`) or deleted (`cleanup --include-ebs`). Reported even when `instances` is empty; in `table` mode orphans render as a second table.

---

## `aws-bootstrap terminate`

Terminate instances created by aws-bootstrap.

```
aws-bootstrap terminate [OPTIONS] [INSTANCE_ID_OR_ALIAS]...
```

Pass instance IDs (e.g. `i-abc123`) or SSH aliases (e.g. `aws-gpu1`) to terminate specific instances. Omit arguments to terminate all aws-bootstrap instances in the region.

**Ownership guard:** explicitly-named instances are checked for the `created-by=aws-bootstrap-g4dn` tag before anything is terminated, and the command fails without terminating if any target lacks it (a wrong id — or the wrong `--region` — would otherwise destroy an unrelated instance). Pass `--force` to override. Terminating *all* instances needs no check: that path already selects by tag.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--yes` / `-y` | flag | false | Skip confirmation prompt (required for `--output json/yaml/table`) |
| `--keep-ebs` | flag | false | Preserve EBS data volumes instead of deleting |
| `--force` | flag | false | Terminate explicitly-named instances even without the `created-by` tag |

### JSON Output

```json
{
  "terminated": [
    {
      "instance_id": "i-0abc123def456",
      "previous_state": "running",
      "current_state": "shutting-down",
      "ssh_alias_removed": "aws-gpu1",
      "ebs_volumes_deleted": ["vol-0abc123"]
    }
  ]
}
```

Fields `ssh_alias_removed` and `ebs_volumes_deleted` are conditional. With `--keep-ebs`, volumes are preserved and not listed in `ebs_volumes_deleted`.

---

## `aws-bootstrap cleanup`

Sync `~/.ssh/config` with live instances **across all enabled regions**. Removes stale aliases (instances gone everywhere); with `--sync` also adds aliases for live instances missing one and repairs aliases whose IP drifted. Never terminates instances. **Conservative guard:** if any region query fails, stale aliases are reported but not removed (`regions_failed` is non-empty).

```
aws-bootstrap cleanup [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dry-run` | flag | false | Preview (diagnostic) — show stale/missing/drifted/orphaned, change nothing |
| `--yes` / `-y` | flag | false | Skip confirmation prompt (required in structured output modes) |
| `--sync` | flag | false | Also add SSH aliases for live instances missing one, and repair drifted IPs |
| `--key-path` | path | `~/.ssh/id_ed25519.pub` | Local SSH public key for aliases added by `--sync` |
| `--ssh-user` | string | `ubuntu` | Remote SSH user for aliases added by `--sync` |
| `--include-ebs` | flag | false | Also find and delete orphan EBS data volumes (across the scanned regions) |

There is no `--region` flag — `cleanup` always scans all enabled regions (an SSH alias records no region, so staleness is only safe to judge globally).

### JSON Output

**Dry run** (keys `stale`/`orphan_volumes`):
```json
{
  "regions_queried": ["us-east-1", "us-west-2"],
  "regions_failed": [],
  "stale": [{"instance_id": "i-0abc123", "alias": "aws-gpu1"}],
  "added": [{"instance_id": "i-0def456", "alias": null, "public_ip": "1.2.3.4"}],
  "updated": [{"instance_id": "i-0aaa111", "alias": "aws-gpu2", "public_ip": "5.6.7.8"}],
  "orphan_volumes": [{"volume_id": "vol-0abc123", "size_gb": 96}]
}
```

**Apply** (keys `cleaned`/`deleted_volumes`):
```json
{
  "regions_queried": ["us-east-1"],
  "regions_failed": [],
  "cleaned": [{"instance_id": "i-0abc123", "alias": "aws-gpu1"}],
  "added": [{"instance_id": "i-0def456", "alias": "aws-gpu3", "public_ip": "1.2.3.4"}],
  "updated": [],
  "deleted_volumes": [{"volume_id": "vol-0abc123", "size_gb": 96, "deleted": true}]
}
```

- `added`/`updated` are populated only with `--sync`. In `added`, `alias` is the deterministic `aws-<cluster-id>-<rank>` for cluster-tagged nodes, else `null` (auto-assigned on apply). Drift `--sync` preserves the existing stanza's user/key/port.
- `orphan_volumes`/`deleted_volumes` only with `--include-ebs`; a failed delete is recorded as `{"deleted": false, "error": "..."}`.
- **Conservative guard** (when `regions_failed` is non-empty): no stale aliases are removed and no orphan volumes are deleted (the scan is incomplete). In apply mode the un-removed stale aliases are surfaced under a `skipped` key (instead of `cleaned`), and orphan volumes appear under `orphan_volumes` (not `deleted_volumes`).
- `-o json`/`yaml` return the full multi-section result above (prefer these for programmatic use). `-o table` flattens every section into one action-tagged table (`Action` / `Instance` / `Alias` / `IP-Volume`) so all removals, adds, and repairs are visible at once.

---

## Instance-type CPU architecture

An AMI must match its instance type's CPU architecture. `launch` and `cluster
launch` read the architecture from the instance type (via EC2) and select an AMI
for it, so you do not need to think about this. AWS names the Deep Learning Base
AMI per architecture (`Deep Learning Base OSS …` for x86_64, `Deep Learning ARM64
Base OSS …` for arm64), and the default `--ami-filter`
(`Deep Learning*Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*`) spans both — the
architecture filter picks the build. So an arm64 type such as `g5g` (NVIDIA T4g)
needs no extra flag:

```bash
aws-bootstrap launch --instance-type g5g.xlarge
```

If no image matches, the command exits non-zero with an error naming the
architecture and a region-pinned `list amis --arch <arch> --region <r>` command.
With a **custom** `--ami-filter` the usual cause is a filter written for the other
architecture — drop or fix it (changing region won't help). With the **default**
filter it means the region doesn't publish the image — try another `--region`.

Every other current GPU family (g4dn, g5, g6/g6e/g6f, gr6/gr6f, g7/g7e, p4d,
p5/p5e/p5en, p6-b200/p6-b300) is x86_64.

## `aws-bootstrap list instance-types`

List EC2 instance types matching a family prefix.

```
aws-bootstrap list instance-types [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prefix` | string | `g4dn` | Instance type family prefix. **Prefix match**, so `p6` also matches `p6-b200`/`p6-b300`, `g6` matches g6/g6e/g6f, and `g` matches every G family. |
| `--region` / `-r` | string (repeatable) | env/profile, then `us-west-2` | Region(s) to query. Each record is tagged with its `region`. |

The table includes a **Quota Family** column (`gvt`/`p`/`dl`) — the AWS vCPU quota
family each type draws from. (These group multiple prefixes, e.g. all G/VT types
including `g5` share `gvt`, which is why a suggested `--family` may not match the
`--prefix`.) In text mode the output ends with copy-paste **Next steps**
(`quota show` and `quota request` pinned to the queried region) for the family
derived from `--prefix`; suppressed for non-GPU families.

### JSON Output

```json
[
  {
    "region": "us-west-2",
    "instance_type": "g4dn.xlarge",
    "vcpus": 4,
    "memory_mib": 16384,
    "quota_family": "gvt",
    "efa_supported": false,
    "gpu": "1x T4 (16384 MiB)"
  }
]
```

`efa_supported` comes from EC2 (`NetworkInfo.EfaSupported`): EFA exists beyond the P-series, on the 8xlarge-and-up sizes of most G families. aws-bootstrap doesn't configure EFA yet.

Fractional-GPU types (g6f, gr6f) report their share of one GPU, e.g.
`"gpu": "1/8 L4 (2861 MiB)"`.

`quota_family` is `null` for non-GPU instance types.

---

## `aws-bootstrap list amis`

List available AMIs matching a name pattern. AMI IDs are region-specific, so each
result is labelled with its region.

```
aws-bootstrap list amis [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter` | string | `Deep Learning*Base OSS Nvidia Driver GPU AMI*` | AMI name pattern (default: the Deep Learning Base AMI, every OS and architecture) |
| `--arch` | `x86_64` \| `arm64` | *(all)* | Restrict to one CPU architecture. Results span both by default; the `architecture` field reports each one. |
| `--region` / `-r` | string (repeatable) | env/profile, then `us-west-2` | Region(s) to query. Each record is tagged with its `region`. |

### JSON Output

```json
[
  {
    "region": "us-west-2",
    "image_id": "ami-0abc123",
    "name": "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) ...",
    "creation_date": "2025-01-10",
    "architecture": "x86_64"
  }
]
```

---

## `aws-bootstrap quota show`

Show current spot and on-demand vCPU quota values for GPU instance families.

```
aws-bootstrap quota show [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--family` | `gvt\|p\|dl` | all | Instance family to show |
| `--region` / `-r` | string (repeatable) | env/profile, then `us-west-2` | Region(s) to query. Each quota is tagged with its `region`. |

Supported families: `gvt` (g3, g4dn, g5, g5g, g6, g6e, vt1), `p` (p2, p3, p4d, p4de, p5, p5e, p5en, p6), `dl` (dl1, dl2q).

### JSON Output

Each quota entry includes its `region`.

```json
{
  "quotas": [
    {
      "region": "us-west-2",
      "family": "gvt",
      "quota_type": "spot",
      "quota_code": "L-3819A6DF",
      "quota_name": "All G and VT Spot Instance Requests",
      "value": 4.0
    },
    {
      "family": "gvt",
      "quota_type": "on-demand",
      "quota_code": "L-DB2E81BA",
      "quota_name": "Running On-Demand G and VT instances",
      "value": 0.0
    },
    {
      "family": "p",
      "quota_type": "spot",
      "quota_code": "L-7212CCBC",
      "quota_name": "All P Spot Instance Requests",
      "value": 0.0
    },
    {
      "family": "p",
      "quota_type": "on-demand",
      "quota_code": "L-417A185B",
      "quota_name": "Running On-Demand P instances",
      "value": 0.0
    },
    {
      "family": "dl",
      "quota_type": "spot",
      "quota_code": "L-85EED4F7",
      "quota_name": "All DL Spot Instance Requests",
      "value": 0.0
    },
    {
      "family": "dl",
      "quota_type": "on-demand",
      "quota_code": "L-6E869C2A",
      "quota_name": "Running On-Demand DL instances",
      "value": 0.0
    }
  ]
}
```

---

## `aws-bootstrap quota request`

Request a vCPU quota increase for a GPU instance family, in one or more regions.

```
aws-bootstrap quota request [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--family` | `gvt\|p\|dl` | `gvt` | Instance family |
| `--type` | `spot\|on-demand` | required | Quota type to increase |
| `--desired-value` | float | required | Desired quota value (vCPUs) |
| `--region` / `-r` | string (repeatable) | env/profile, then `us-west-2` | Region(s) to submit the request in. All are validated up front; if any region's current value ≥ `--desired-value`, nothing is submitted. |
| `--yes` / `-y` | flag | false | Skip confirmation prompt (required for `--output json/yaml/table`) |

### JSON Output

Returns one entry per region under `requests` (the shape is a list even for a single region):

```json
{
  "requests": [
    {
      "request_id": "abc123-def456",
      "status": "PENDING",
      "quota_code": "L-3819A6DF",
      "quota_name": "All G and VT Spot Instance Requests",
      "desired_value": 4.0,
      "region": "us-west-2",
      "quota_type": "spot",
      "family": "gvt"
    }
  ]
}
```

Field `case_id` is included on an entry when a support case is opened.

---

## `aws-bootstrap quota history`

Show history of vCPU quota increase requests for GPU instance families.

```
aws-bootstrap quota history [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--family` | `gvt\|p\|dl` | all | Filter by instance family |
| `--type` | `spot\|on-demand` | both | Filter by quota type |
| `--status` | `PENDING\|CASE_OPENED\|APPROVED\|DENIED\|CASE_CLOSED\|NOT_APPROVED` | all | Filter by request status |
| `--region` / `-r` | string (repeatable) | env/profile, then `us-west-2` | Region(s) to query. Each request is tagged with its `region`; results merged and sorted newest-first. |

### JSON Output

Each request includes its `region`.

```json
{
  "requests": [
    {
      "request_id": "abc123-def456",
      "status": "APPROVED",
      "region": "us-west-2",
      "family": "gvt",
      "quota_type": "spot",
      "quota_code": "L-3819A6DF",
      "quota_name": "All G and VT Spot Instance Requests",
      "desired_value": 4.0,
      "created": "2025-06-01T00:00:00+00:00"
    }
  ]
}
```

Requests are sorted newest-first. Field `case_id` is included when a support case is associated.

---

## `aws-bootstrap cluster launch`

Launch (or incrementally grow) a multi-node training cluster: N GPU instances tagged with a shared `--cluster-id`, all in one AZ inside a cluster placement group, with a self-referencing security-group rule for intra-cluster NCCL/rendezvous traffic. Re-run with a higher `--nodes` to add nodes toward the target.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | (required) | Cluster identifier (EC2 tag; tags are the source of truth) |
| `--nodes` | int | `2` | Target number of nodes |
| `--instance-type` | string | `g4dn.xlarge` | EC2 instance type |
| `--spot/--on-demand` | flag | spot | Pricing |
| `--region` | string | resolved | AWS region (a single AZ is chosen within it) |
| `--key-path` | path | `~/.ssh/id_ed25519.pub` | Local SSH public key (auto-generated if absent) |
| `--volume-size` | int | `100` | Root EBS volume size (GB, gp3) |
| `--no-setup` | flag | false | Skip remote setup |
| `--ssh-port` | int | `22` | SSH port |
| `--python-version` | string | none | Python version for remote venv |
| `--wait` | flag | false | On spot capacity shortfall, retry each node with backoff until `--wait-timeout` (no on-demand prompt/fallback) |
| `--wait-timeout` | duration | `30m` | Max wait per node for spot capacity when `--wait` is set (e.g. `30m`, `1h`) |
| `--availability-zone` / `--az` | string | none | AZ for a **new** cluster (implies its region); an existing cluster stays in its AZ (a conflicting value is an error) |
| `--cheapest-az` | flag | false | Spot, new cluster: the first node tries the region's AZs cheapest-first; the rest follow it |
| `--skip-key-check` | flag | false | Skip the pre-launch SSH key check (runs before any AWS change) |
| `--iam-instance-profile` | string | none | Existing IAM instance profile (name or ARN) for the new nodes; growing a cluster whose nodes have a different one warns |

### JSON Output

```json
{
  "cluster_id": "ml1",
  "region": "us-east-1",
  "availability_zone": "us-east-1c",
  "placement_group": "aws-bootstrap-cluster-ml1",
  "nodes_added": 4,
  "node_count": 4,
  "nodes": [
    {"rank": 0, "instance_id": "i-0abc", "public_ip": "1.2.3.4", "alias": "aws-ml1-0"}
  ]
}
```

When the cluster already has `--nodes` nodes, `nodes_added` is `0` and no instances are launched.

---

## `aws-bootstrap cluster status`

Show a cluster's nodes (rank, state, type, AZ, IPs). Omit `--cluster-id` to list all clusters in the region.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | none | Cluster id (omit to list all clusters) |
| `--region` | string | resolved | AWS region |

### JSON Output

With `--cluster-id`:

```json
{
  "cluster_id": "ml1",
  "region": "us-east-1",
  "nodes": [
    {"rank": 0, "instance_id": "i-0abc", "state": "running", "instance_type": "g5.xlarge",
     "az": "us-east-1c", "public_ip": "1.2.3.4", "private_ip": "10.0.0.5"}
  ]
}
```

Without `--cluster-id`: `{"region": "...", "clusters": [{"cluster_id": "ml1", "node_count": 4}]}`.

---

## `aws-bootstrap cluster prepare`

Verify a cluster and run a distributed canary. Checks each node is reachable and has a GPU, that CUDA versions are consistent across nodes (fails fast on skew), writes a per-node `~/.aws-bootstrap-cluster` config (the `AWSB_*` env contract), then runs the canary unless `--no-canary`.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | (required) | Cluster id |
| `--region` | string | resolved | AWS region |
| `--key-path` | path | `~/.ssh/id_ed25519.pub` | Local SSH public key |
| `--ssh-user` | string | `ubuntu` | Remote SSH user |
| `--ssh-port` | int | `22` | SSH port |
| `--no-canary` | flag | false | Skip the auto-canary |

### JSON Output

```json
{"cluster_id": "ml1", "verified": true, "canary_passed": true, "master_addr": "10.0.0.5", "node_count": 4}
```

Exits non-zero if a node is unreachable/GPU-less, CUDA versions are inconsistent, or the canary fails.

---

## `aws-bootstrap cluster test`

Run the built-in distributed canary across the cluster (a re-runnable heartbeat). The canary runs the same `torchrun` command on every node (c10d rendezvous, endpoint = rank-0 private IP).

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | (required) | Cluster id |
| `--region` | string | resolved | AWS region |
| `--key-path` | path | `~/.ssh/id_ed25519.pub` | Local SSH public key |
| `--ssh-user` | string | `ubuntu` | Remote SSH user |
| `--ssh-port` | int | `22` | SSH port |

### JSON Output

```json
{"cluster_id": "ml1", "passed": true,
 "results": [{"rank": 0, "instance_id": "i-0abc", "returncode": 0}]}
```

Exits non-zero if any node's canary returns non-zero.

---

## `aws-bootstrap cluster run`

Distribute a training `SCRIPT` to every node and run it across the cluster. A `.py` script is launched with `torchrun` (c10d rendezvous, endpoint = rank-0 private IP); a `.sh` script runs as-is with the `AWSB_*` env contract exported. With `--data-script`, a prep script runs on every node first (a barrier; training won't start until all preps succeed). Per-node logs are written under `--log-dir`.

### Usage

```bash
aws-bootstrap cluster run --cluster-id ml1 [--data-script prep.sh] train.py -- --epochs 5
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | (required) | Cluster id |
| `SCRIPT` | path (arg) | (required) | Local training script (`.py` → torchrun; `.sh` → env escape hatch) |
| `ARGS` | passthrough | none | Args after `--` are passed to the script |
| `--data-script` | path | none | Script copied + run on each node before training (idempotent) |
| `--nproc-per-node` | int | `1` | Processes (GPUs) per node |
| `--log-dir` | path | `.aws-bootstrap/clusters` | Local dir for per-node logs (`<log-dir>/<cluster-id>/rank<N>.log`) |
| `--region` / `--profile` / `--key-path` / `--ssh-user` / `--ssh-port` | | | Standard connection options |

### Env contract (`.sh` escape hatch)

Each node has these exported: `AWSB_CLUSTER_ID`, `AWSB_NODE_RANK`, `AWSB_NUM_NODES`, `AWSB_NUM_GPUS_PER_NODE`, `AWSB_NODE_IPS` (newline-separated, rank order), `AWSB_MASTER_ADDR` (rank-0 private IP).

### JSON Output

```json
{"cluster_id": "ml1", "succeeded": true, "log_dir": ".aws-bootstrap/clusters/ml1",
 "results": [{"rank": 0, "instance_id": "i-0abc", "returncode": 0}]}
```

Exits non-zero if any node returns non-zero (or if SCP/data-prep fails).

---

## `aws-bootstrap cluster terminate`

Terminate all nodes of a cluster, remove their SSH aliases, wait for full termination, then delete the placement group. (`cluster launch` does not create per-node EBS data volumes yet, so there are none to preserve — there is no `--keep-ebs` here.)

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | (required) | Cluster id to terminate |
| `--region` | string | resolved | AWS region |
| `--yes` | flag | false | Skip confirmation (required in structured output modes) |

### JSON Output

```json
{"cluster_id": "ml1", "region": "us-east-1", "terminated": ["i-0abc", "i-1def"]}
```

---

## Notes

- **Cluster execution commands** (`prepare`, `test`, `run`) require **every** node to be `running`; they fail with an actionable error if any node is `stopped`/`shutting-down` (a distributed job needs all ranks at the rendezvous, and a reclaimed rank 0 would hang the whole job). `cluster status` still reports every non-terminated node — it is the diagnostic view.
- **SSH aliases** use sequential numbering (`aws-gpu1`, `aws-gpu2`, etc.) and are managed in `~/.ssh/config`; cluster nodes use deterministic `aws-<cluster-id>-<rank>` aliases (e.g. `aws-ml1-0`)
- **EBS volumes** are tagged with `created-by=aws-bootstrap-g4dn` for automatic discovery
- **Spot capacity**: on a fully-exhausted spot sweep (`InsufficientInstanceCapacity` in every `--region`) **without `--wait`**, the launcher offers the on-demand fallback (auto-confirmed in structured modes). With `--wait` it retries with backoff and hard-fails on timeout (never auto-buys on-demand). Quota errors and `SpotMaxPriceTooLow` are **not** auto-fallback triggers — in multi-region mode they warn and skip to the next region, hard-failing only when every region is blocked.
- **Remote setup** installs CUDA-matched PyTorch, Jupyter, GPU benchmark, and VSCode CUDA debug configs
- The default AMI filter targets the Ubuntu 24.04 Deep Learning Base AMI (OSS NVIDIA driver) for the instance type's architecture — the same image as AWS's SSM "latest DLAMI" parameter. Ubuntu 26.04 builds exist too: `--ami-filter "Deep Learning*Base OSS Nvidia Driver GPU AMI (Ubuntu 26.04)*"`
