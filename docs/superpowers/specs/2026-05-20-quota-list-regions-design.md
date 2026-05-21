# Design: region-aware `quota` and `list` commands

**Date:** 2026-05-20
**Scope:** `quota show`, `quota history`, `quota request`, `list instance-types`, `list amis`
**Lands in:** PR #30 (alongside multi-region `status`)

## Problem

Quotas and AMIs are region-scoped (AMI IDs even differ per region), but
`quota show`/`history` and `list instance-types`/`amis` don't say which region
they operate on, and `--region` accepts only a single value. Users can't tell
which region's data they're seeing, nor query several at once.

## Decisions

- **Default region:** the single resolved region (explicit `--region` →
  `AWS_DEFAULT_REGION`/profile → `us-west-2`), always **labeled** in output. Not
  all-enabled-regions (noisy for quotas/AMIs, unsafe for `quota request`).
- **`--region` is repeatable** (`multiple=True`, `-r`) on all five commands,
  resolved via the existing `cli.resolve_region_list(region, profile)` helper.
- **`quota request` submits per region**, validating all regions up front.
- Per-region iteration is **sequential** (region counts are small); errors
  propagate (no status-style warn-and-continue — a possible later enhancement).

## Behavior

| Invocation | Regions | Header |
|------------|---------|--------|
| `<cmd>` (no `--region`) | single resolved | `Region: <region>` |
| `<cmd> -r a -r b` | `a`, `b` (in order) | grouped per region |

Each structured record gains a `"region"` field; tables gain a `Region` column.
Records are built fresh with the region rather than mutating helper results.

### Per command
- **`quota show`** — single region: `Region: <r>` line then the listing. Multiple:
  a `EC2 GPU vCPU Quotas — <r>:` block per region. The "To request an increase"
  hint stays pinned to that region (`--region <r>`).
- **`quota history`** — tag each request with `region`, merge, sort newest-first;
  show a `Region` line per request and list queried regions in the header.
- **`quota request`** — resolve regions; fetch each region's current value; if any
  region's current ≥ desired, abort naming those regions (no partial submit).
  Confirmation lists all target regions. Submit per region.
  **Structured output changes** from a single object to
  `{"requests": [{request_id, status, region, family, quota_type, case_id?}, …]}`.
- **`list instance-types`** / **`list amis`** — single region: `Region: <r>` then the
  listing. Multiple: a per-region block. Each record tagged with `region`
  (critical for AMIs, whose `image_id` differs per region).

## Implementation

`cli.py`:
- Switch the `--region` option on the five commands to `multiple=True` with `-r`.
- `regions = resolve_region_list(region, profile)` (returns a non-empty ordered
  tuple).
- Build one region-scoped boto3 client per region inside the per-region loop.
- Factor each command's text rendering into a small inner `render(region, items)`
  so single- and multi-region paths share it.
- Structured: accumulate `{"region": r, **record}` across regions, emit once with
  a `Region` column added to headers.

No changes needed in `quota.py` / `ec2.py` helpers — they already take a client
and are region-agnostic.

## Testing (idiomatic pytest)

- Add `aws_bootstrap/tests/conftest.py` with shared fixtures: `runner`
  (`CliRunner`), a `mock_cli_session` fixture patching `cli.boto3.Session`, and
  data fixtures (`quota_rows`, `instance_type_rows`, `ami_rows`).
- Use `pytest.mark.parametrize` for single-region vs multi-region cases and per
  command.
- Cover: region label in header (single); repeatable `--region` queries each and
  tags records; `quota request` aborts when current ≥ desired and submits per
  region otherwise; JSON includes `region`; help shows repeatable `--region`.

## Docs

README.md, CLAUDE.md, `docs/` as needed, and the agent skill
(`SKILL.md`, `references/commands.md`) — region-aware `list`/`quota`, repeatable
`--region`, and the new `quota request` JSON shape.
