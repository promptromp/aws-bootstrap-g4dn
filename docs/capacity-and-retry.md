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

### Fail-fast errors (never retried)

| Error | Why no retry |
|-------|--------------|
| `VcpuLimitExceeded` / `MaxSpotInstanceCountExceeded` | Account quota — see `aws-bootstrap quota` |
| `SpotMaxPriceTooLow` | Bid below market; waiting won't change it — use `--on-demand` |

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
