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

The active region(s) are shown in `launch`, `status`, `terminate`, `cleanup`, `quota`, and `list` output (single region → a `Region:` line; multiple → a per-region block).

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
| `--ebs-volume-id` | string | none | Attach existing EBS volume (at /data); pins instance to the volume's AZ |

`--ebs-storage` and `--ebs-volume-id` are mutually exclusive.

`--ebs-volume-id` automatically pins the instance to the volume's availability zone (EBS volumes are AZ-scoped). The launch therefore targets the volume's region, and spot capacity is limited to that one AZ — combine with `--wait` if the AZ is temporarily short on capacity. If the volume isn't found in the target region, the launch fails fast with a region-named error before provisioning anything.

**`--wait` + multiple `--region`:** a region sweep is the inner loop, backoff is the outer loop. Each cycle tries spot in every `--region` in order with no delay between regions; only when *all* regions miss does it sleep (capped+jittered exponential backoff, escalating per sweep) and sweep again, until `--wait-timeout` total wall-clock, then hard-fail. So `--wait --region A --region B` = "try A then B instantly; if both dry, back off and retry both" — not "wait on A then try B." Region order wins every tie. Without `--wait`, exactly one sweep then on-demand fallback.

### JSON Output

**Normal launch:**
```json
{
  "instance_id": "i-0abc123def456",
  "public_ip": "54.200.1.2",
  "instance_type": "g4dn.xlarge",
  "availability_zone": "us-west-2a",
  "ami_id": "ami-0abc123",
  "pricing": "spot",
  "region": "us-west-2",
  "regions_tried": ["us-west-2", "us-east-1"],
  "ssh_alias": "aws-gpu1",
  "cuda_version": "13.2",
  "ebs_volume": {
    "volume_id": "vol-0abc123",
    "mount_point": "/data",
    "size_gb": 96
  }
}
```

The `ebs_volume` field is only present when `--ebs-storage` or `--ebs-volume-id` is used. `cuda_version` is present only when remote setup ran and a CUDA version was detected (omitted with `--no-setup` or on non-CUDA/smoke-test instances). `region` is the region the instance actually launched in; `regions_tried` lists all regions attempted in order.

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
  "wait_timeout_seconds": 1800
}
```

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
  ]
}
```

Top-level `regions_queried` lists every region the query was attempted against (including any that subsequently failed). `regions_failed` is present only when one or more regions could not be queried (e.g. unauthorized); those regions appear in both lists. Per-instance fields `gpu`, `ebs_volumes`, `spot_price_per_hour`, `uptime_seconds`, and `estimated_cost` are conditional.

---

## `aws-bootstrap terminate`

Terminate instances created by aws-bootstrap.

```
aws-bootstrap terminate [OPTIONS] [INSTANCE_ID_OR_ALIAS]...
```

Pass instance IDs (e.g. `i-abc123`) or SSH aliases (e.g. `aws-gpu1`) to terminate specific instances. Omit arguments to terminate all aws-bootstrap instances in the region.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--yes` / `-y` | flag | false | Skip confirmation prompt (required for `--output json/yaml/table`) |
| `--keep-ebs` | flag | false | Preserve EBS data volumes instead of deleting |

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

Remove stale SSH config entries for terminated instances.

```
aws-bootstrap cleanup [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dry-run` | flag | false | Preview removals without modifying anything |
| `--yes` / `-y` | flag | false | Skip confirmation prompt |
| `--include-ebs` | flag | false | Also find and delete orphan EBS data volumes |

### JSON Output

**Dry run:**
```json
{
  "stale": [
    {"instance_id": "i-0abc123", "alias": "aws-gpu1"}
  ],
  "dry_run": true,
  "orphan_volumes": [
    {"volume_id": "vol-0abc123", "size_gb": 96, "instance_id": "i-0abc123"}
  ]
}
```

`orphan_volumes` is only present when `--include-ebs` is used.

**Actual cleanup:**
```json
{
  "cleaned": [
    {"instance_id": "i-0abc123", "alias": "aws-gpu1", "removed": true}
  ],
  "deleted_volumes": [
    {"volume_id": "vol-0abc123", "size_gb": 96, "deleted": true}
  ]
}
```

`deleted_volumes` is only present when `--include-ebs` is used.

---

## `aws-bootstrap list instance-types`

List EC2 instance types matching a family prefix.

```
aws-bootstrap list instance-types [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prefix` | string | `g4dn` | Instance type family prefix to filter on |
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
    "gpu": "1x T4 (16384 MiB)"
  }
]
```

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
| `--filter` | string | `Deep Learning Base OSS Nvidia Driver GPU AMI*` | AMI name pattern |
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

## `aws-bootstrap cluster terminate`

Terminate all nodes of a cluster, remove their SSH aliases, and delete the placement group.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cluster-id` | string | (required) | Cluster id to terminate |
| `--region` | string | resolved | AWS region |
| `--keep-ebs` | flag | false | Preserve per-node EBS data volumes |
| `--yes` | flag | false | Skip confirmation (required in structured output modes) |

### JSON Output

```json
{"cluster_id": "ml1", "region": "us-east-1", "terminated": ["i-0abc", "i-1def"]}
```

---

## Notes

- **SSH aliases** use sequential numbering (`aws-gpu1`, `aws-gpu2`, etc.) and are managed in `~/.ssh/config`; cluster nodes use deterministic `aws-<cluster-id>-<rank>` aliases (e.g. `aws-ml1-0`)
- **EBS volumes** are tagged with `created-by=aws-bootstrap-g4dn` for automatic discovery
- **Spot capacity**: on a fully-exhausted spot sweep (`InsufficientInstanceCapacity` in every `--region`) **without `--wait`**, the launcher offers the on-demand fallback (auto-confirmed in structured modes). With `--wait` it retries with backoff and hard-fails on timeout (never auto-buys on-demand). Quota errors and `SpotMaxPriceTooLow` are **not** auto-fallback triggers — in multi-region mode they warn and skip to the next region, hard-failing only when every region is blocked.
- **Remote setup** installs CUDA-matched PyTorch, Jupyter, GPU benchmark, and VSCode CUDA debug configs
- The default AMI filter targets Ubuntu 24.04 Deep Learning AMIs with the OSS NVIDIA driver
