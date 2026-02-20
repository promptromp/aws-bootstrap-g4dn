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
| `--region` | string | `us-west-2` | AWS region |
| `--profile` | string | `AWS_PROFILE` env | AWS profile override |

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
| `--key-path` | path | `~/.ssh/id_ed25519.pub` | SSH public key path |
| `--key-name` | string | `aws-bootstrap-key` | AWS key pair name |
| `--security-group` | string | `aws-bootstrap-ssh` | Security group name |
| `--volume-size` | int | `100` | Root EBS volume size (GB, gp3) |
| `--no-setup` | flag | false | Skip remote setup script |
| `--dry-run` | flag | false | Validate without launching |
| `--python-version` | string | none | Python version for remote venv (e.g. 3.13) |
| `--ssh-port` | int | `22` | SSH port for security group and connection |
| `--ebs-storage` | int | none | Create new EBS data volume (GB, gp3, at /data) |
| `--ebs-volume-id` | string | none | Attach existing EBS volume (at /data) |

`--ebs-storage` and `--ebs-volume-id` are mutually exclusive.

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
  "ssh_alias": "aws-gpu1",
  "ebs_volume": {
    "volume_id": "vol-0abc123",
    "mount_point": "/data",
    "size_gb": 96
  }
}
```

The `ebs_volume` field is only present when `--ebs-storage` or `--ebs-volume-id` is used.

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
  "region": "us-west-2"
}
```

---

## `aws-bootstrap status`

Show running instances created by aws-bootstrap.

```
aws-bootstrap status [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gpu` | flag | false | Query GPU info (CUDA, driver) via SSH |
| `--instructions` / `--no-instructions` / `-I` | flag | true | Show connection commands for running instances |

### JSON Output

```json
{
  "instances": [
    {
      "instance_id": "i-0abc123def456",
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
  ]
}
```

Fields `gpu`, `ebs_volumes`, `spot_price_per_hour`, `uptime_seconds`, and `estimated_cost` are conditional.

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

### JSON Output

```json
[
  {
    "instance_type": "g4dn.xlarge",
    "vcpus": 4,
    "memory_mib": 16384,
    "gpu": "1x T4 (16384 MiB)"
  }
]
```

---

## `aws-bootstrap list amis`

List available AMIs matching a name pattern.

```
aws-bootstrap list amis [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter` | string | `Deep Learning Base OSS Nvidia Driver GPU AMI*` | AMI name pattern |

### JSON Output

```json
[
  {
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

Supported families: `gvt` (g3, g4dn, g5, g5g, g6, g6e, vt1), `p` (p2, p3, p4d, p4de, p5, p5e, p5en, p6), `dl` (dl1, dl2q).

### JSON Output

```json
{
  "quotas": [
    {
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

Request a vCPU quota increase for a GPU instance family.

```
aws-bootstrap quota request [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--family` | `gvt\|p\|dl` | `gvt` | Instance family |
| `--type` | `spot\|on-demand` | required | Quota type to increase |
| `--desired-value` | float | required | Desired quota value (vCPUs) |
| `--yes` / `-y` | flag | false | Skip confirmation prompt (required for `--output json/yaml/table`) |

### JSON Output

```json
{
  "request_id": "abc123-def456",
  "status": "PENDING",
  "quota_code": "L-3819A6DF",
  "quota_name": "All G and VT Spot Instance Requests",
  "desired_value": 4.0,
  "quota_type": "spot",
  "family": "gvt"
}
```

Field `case_id` is included when a support case is opened.

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

### JSON Output

```json
{
  "requests": [
    {
      "request_id": "abc123-def456",
      "status": "APPROVED",
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

## Notes

- **SSH aliases** use sequential numbering (`aws-gpu1`, `aws-gpu2`, etc.) and are managed in `~/.ssh/config`
- **EBS volumes** are tagged with `created-by=aws-bootstrap-g4dn` for automatic discovery
- **Spot pricing** auto-falls back to on-demand on `InsufficientInstanceCapacity` or `SpotMaxPriceTooLow` errors
- **Remote setup** installs CUDA-matched PyTorch, Jupyter, GPU benchmark, and VSCode CUDA debug configs
- The default AMI filter targets Ubuntu 24.04 Deep Learning AMIs with the OSS NVIDIA driver
