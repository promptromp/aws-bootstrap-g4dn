# Design: Multi-region `status` command

**Date:** 2026-05-20
**Scope:** `aws-bootstrap status` only

## Problem

`aws-bootstrap status` is hard-coded to a single `--region` (default `us-west-2`).
The boto3 EC2 client is region-bound, so `find_tagged_instances` only ever sees
one region. If a user launched instances in another region, naked `status`
reports "No active aws-bootstrap instances found" — which is misleading, because
the instances exist, just not in the default region.

## Goal

Make naked `aws-bootstrap status` show **all** live aws-bootstrap instances
across all enabled regions, labelling each instance with its region. When one or
more `--region` values are supplied, restrict the query to exactly those regions
and say so in the output header.

## Behavior

| Invocation | Regions queried | Header (text mode) |
|------------|-----------------|--------------------|
| `status` | all account-enabled regions | `Querying N enabled region(s): ...` |
| `status --region us-east-1` | `us-east-1` only | `Showing status for selected region(s): us-east-1` |
| `status --region us-east-1 --region eu-west-1` | both | `Showing status for selected region(s): us-east-1, eu-west-1` |

- `--region` becomes a Click `multiple=True` option (default: empty tuple).
- Region discovery (no `--region`): `ec2.describe_regions(AllRegions=False)`,
  which returns regions the account has opted into. Sorted alphabetically.
- Per-region queries run in parallel via `ThreadPoolExecutor` (~10 workers).
- **Error tolerance:** if querying a region fails (e.g. `AuthFailure` in an
  opt-in region, throttling), emit a stderr warning, skip that region, and
  continue. The command exits 0 if at least one region was queried.

## Output

### Text mode
- Header line per the table above.
- Each instance gains a `Region:` line (placed right after the instance-id/state
  line, before `Type`).
- Instances are grouped/sorted by region, then by launch time, for stable
  output.
- The closing "To terminate" hint includes `--region <region>` for the first
  instance (since terminate is still single-region).
- A `regions failed` summary line is printed (stderr warnings already shown
  inline) only when at least one region failed.

### Structured mode (json/yaml/table)
- Each instance record gains `"region": "<region>"`.
- Top-level result gains:
  - `"regions_queried": ["us-east-1", ...]`
  - `"regions_failed": [{"region": "...", "error": "..."}]` — included only when
    non-empty.
- Table headers add a `Region` column.

## Implementation

### `aws_bootstrap/ec2.py`
- `list_enabled_regions(ec2_client) -> list[str]` — wraps
  `describe_regions(AllRegions=False)`, returns sorted region names.
- `find_tagged_instances_in_regions(session, tag_value, regions, max_workers=10)
  -> tuple[list[dict], list[dict]]` — for each region, builds a region-bound
  client, calls the existing `find_tagged_instances`, attaches `Region` to each
  instance dict. Returns `(instances, failures)` where `failures` is a list of
  `{"region": ..., "error": ...}`. Runs region queries in a thread pool.
  - Each returned instance dict gains a `"Region"` key.

### `aws_bootstrap/cli.py` — `status`
- `--region` → `multiple=True`, no default.
- Build a base session/client (in the historical default region, or the first
  supplied region) solely to call `describe_regions` when needed.
- Resolve the region list:
  - if `region` tuple is non-empty → use it (selected mode);
  - else → `list_enabled_regions(...)` (all mode).
- Print the appropriate header.
- Call `find_tagged_instances_in_regions`.
- Maintain a `region → ec2_client` cache so per-instance calls
  (`get_spot_price`, `find_ebs_volumes_for_instance`) hit the right region.
- Emit per-region failure warnings.
- Add `region` to each structured record, plus `regions_queried` /
  `regions_failed` at the top level.

The per-instance display loop is otherwise unchanged; it just reads
`inst["Region"]` and selects the right client from the cache.

## Out of scope (noted as follow-up)

`terminate` and `cleanup` keep their single `--region` default. A related wart:
after `status` shows an instance in a non-default region, the user must pass
`--region <that-region>` to `terminate` it. Resolving that (e.g. recording region
in the SSH-config block and auto-detecting it) is a separate change.

## Testing

- `tests/test_ec2.py`:
  - `list_enabled_regions` parses `describe_regions` output and sorts.
  - `find_tagged_instances_in_regions` aggregates across regions, attaches
    `Region`, and captures failures without raising.
- `tests/test_cli.py` (`CliRunner`, mocked boto3 + ec2 helpers):
  - naked `status` queries all enabled regions and shows region labels.
  - `status --region a --region b` queries only those, prints selected-region
    header.
  - a failing region produces a warning and the command still exits 0 with the
    surviving region's instances.
  - JSON output includes `region` per instance and `regions_queried`.
