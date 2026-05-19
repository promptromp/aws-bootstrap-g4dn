# Capacity, Multi-Region & `--wait` Retry

GPU instances — especially spot `g4dn`, `g5`, `g6`, and `p`-family — are frequently
unavailable in a given region/AZ. This document explains why, and how
`aws-bootstrap launch` finds capacity for you.

## Why capacity errors happen

`run_instances` returns `InsufficientInstanceCapacity` when AWS has no spare
capacity for the requested instance type **in that region and availability
zone** at that moment. Key properties:

- **It is region- and AZ-scoped.** `g4dn.xlarge` spot may be exhausted in
  `us-west-2` while plentiful in `us-east-1` (and vice versa, minutes later).
- **It is transient.** Capacity frees up continuously as other accounts'
  instances stop/terminate. Retrying the same region a few minutes later
  often succeeds.
- **It is not a quota problem.** Quota errors (`VcpuLimitExceeded`,
  `MaxSpotInstanceCountExceeded`) are a different failure — your account limit,
  not AWS capacity. Those never resolve by waiting and **fail fast**.

## Strategies

### Multiple regions (`--region` repeated)

```bash
aws-bootstrap launch --region us-west-2 --region us-east-1 --region eu-west-1
```

Each launch attempt iterates the regions **in the order you list them**,
spot-first, and uses the first region that has capacity. Region-scoped
prerequisites (AMI lookup, key-pair import, security group) are prepared once
per region and cached across retries.

### Bounded-backoff wait (`--wait` / `--wait-timeout`)

```bash
aws-bootstrap launch --wait --wait-timeout 30m
aws-bootstrap launch --region us-west-2 --region us-east-1 --wait --wait-timeout 1h
```

With `--wait`, exhausting all regions on spot does **not** fail. Instead the
command sleeps and retries until `--wait-timeout`.

- **Backoff schedule:** exponential — `30s, 60s, 120s, 240s, …` — capped at
  **300s**, with **±20% jitter** so concurrent users don't synchronize. The
  final sleep is clamped so it never overshoots the deadline.
- **Heartbeat:** each wait cycle prints
  `[wait] cycle N: no capacity in <regions> — next attempt in <s>s (elapsed <s>s)`.
- **On timeout:** the command **hard-fails** with a clear error. It does *not*
  silently fall back to on-demand — choose `--on-demand` explicitly if you want
  guaranteed (paid) capacity.
- **`--wait-timeout` format:** `90s`, `30m`, `1h`, or a bare integer (seconds).
  Default `30m`.

### How `--wait` and multiple `--region` combine

This is the most important thing to understand. **A region sweep is the inner
loop; backoff is the outer loop:**

```
repeat until --wait-timeout:                 # outer loop — backoff between sweeps
    for region in --region order:            # inner loop — NO delay between regions
        try spot in region
        capacity?  -> launch here, done
        no capacity -> immediately try next region
    every region missed -> sleep (backoff), then sweep again from the top
```

Consequences:

- `--wait --region A --region B` does **not** mean "wait on A, then try B."
  It means: try A then B back-to-back (no sleep between them); if **both** are
  dry, back off, then try A then B **again** — repeating until the timeout.
- **Backoff is per full sweep, not per region.** The sleep means "all my
  regions are dry right now." The interval escalates per sweep
  (`30s, 60s, 120s, …` capped at 300s), independent of how many regions you
  listed.
- **Region order still wins every tie.** In each sweep the first region with
  capacity gets the instance, so list your most-preferred region first.
- **`--wait-timeout` is total wall-clock** across all sweeps (sleeps + the
  brief sweep attempts), not per region or per sweep. The last sleep is
  clamped so it never overshoots the deadline.
- Adding more regions widens each sweep (more chances per cycle) but does
  **not** change the backoff schedule or the total timeout.

Example — `--region us-east-1 --region us-west-2 --wait --wait-timeout 1h`:
sweep both regions instantly; if both dry, sleep ~30s; sweep both again;
sleep ~60s; … capped at ~300s between sweeps; the moment either region has
spot capacity, launch there; if 1h elapses with no capacity in either,
hard-fail.

### Region-fatal errors (never *waited*, but the next region is still tried)

| Error | Behavior |
|-------|----------|
| `VcpuLimitExceeded` / `MaxSpotInstanceCountExceeded` (quota) | Account quota is **per region**. The launcher prints a `WARNING` for that region (with a region-pinned `aws-bootstrap quota show` / `quota request` hint), drops the region, and **moves on to the next `--region`**. It never triggers a `--wait` sleep (quota won't free up by waiting). |
| `SpotMaxPriceTooLow` | Same handling: warn, drop the region, try the next one (spot price differs per region). |

If **every** region is quota/price-blocked (and none have retryable
capacity errors), the command **fails hard** with an aggregated message
listing each region's reason and the full, region-pinned remediation hint
for every quota/price region. The per-region `WARNING`s are emitted *as they
happen* (before the next region is tried), so you see the quota hint for a
region even if a later region ultimately succeeds.

> Run the suggested `aws-bootstrap quota …` commands verbatim — they are
> already pinned to `--region <the region that failed>`; don't let them
> resolve to your default region.

### On-demand fallback (without `--wait`)

A single exhausted spot pass (no `--wait`) still offers the interactive
"Retry as on-demand instance?" prompt, which then tries on-demand across all
listed regions. In structured output modes (`-o json/yaml/table`) this is
auto-confirmed.

## Region default precedence

`--region` resolution (all commands):

1. Explicit `--region` flag(s), in order.
2. `AWS_DEFAULT_REGION` / the active profile's configured region.
3. `us-west-2` (final fallback).

> **Behavior change:** earlier versions hardcoded `us-west-2` and ignored the
> profile/env region. A profile configured for `us-east-1` now defaults to
> `us-east-1`. The resolved/active region is shown in command output.

## Recommended region lists

GPU spot availability shifts constantly, but these region sets tend to have
the deepest pools per family (try in roughly this order, adjust for data
locality and latency):

| Family | Suggested `--region` order |
|--------|-----------------------------|
| `g4dn` (T4) | `us-east-1 us-west-2 us-east-2 eu-west-1` |
| `g5` / `g6` (A10G / L4) | `us-east-1 us-west-2 us-east-2 ap-south-1` |
| `p3` / `p4` / `p5` | `us-east-1 us-east-2 us-west-2` |

Combine with `--wait` for unattended provisioning:

```bash
aws-bootstrap launch --instance-type g5.xlarge \
  --region us-east-1 --region us-west-2 --region us-east-2 \
  --wait --wait-timeout 45m
```
