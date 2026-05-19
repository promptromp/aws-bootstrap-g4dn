"""Region resolution and bounded-backoff helpers for capacity retries.

Pure, side-effect-free functions so the retry policy is unit-testable
without touching AWS or the clock.
"""

from __future__ import annotations
import random
import re

from .config import DEFAULT_REGION


# Backoff defaults (seconds). Exponential, capped, with jitter.
BACKOFF_BASE = 30.0
BACKOFF_CAP = 300.0
BACKOFF_FACTOR = 2.0
BACKOFF_JITTER = 0.2  # +/- fraction applied to each interval

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smh]?)\s*$", re.IGNORECASE)
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600}


def resolve_regions(explicit: tuple[str, ...], session_region: str | None) -> tuple[str, ...]:
    """Resolve the ordered region list.

    Precedence: explicit ``--region`` flags (order preserved) >
    the boto3 session region (``AWS_DEFAULT_REGION`` / profile config) >
    the hardcoded default.
    """
    if explicit:
        return tuple(explicit)
    if session_region:
        return (session_region,)
    return (DEFAULT_REGION,)


def parse_duration(value: str) -> int:
    """Parse a duration like ``30m``, ``90s``, ``1h`` or a bare ``3600`` into seconds.

    Bare integers are interpreted as seconds. Raises ``ValueError`` on
    malformed input or non-positive durations.
    """
    match = _DURATION_RE.match(value)
    if not match:
        raise ValueError(f"Invalid duration: {value!r} (expected e.g. '30m', '90s', '1h', or seconds)")
    amount = int(match.group(1)) * _DURATION_UNITS[match.group(2).lower()]
    if amount <= 0:
        raise ValueError(f"Duration must be positive: {value!r}")
    return amount


def backoff_sleep_seconds(
    attempt: int,
    *,
    base: float = BACKOFF_BASE,
    cap: float = BACKOFF_CAP,
    factor: float = BACKOFF_FACTOR,
    jitter: float = BACKOFF_JITTER,
    rng: random.Random | None = None,
) -> float:
    """Sleep duration before retry ``attempt`` (0-based, i.e. after the first miss).

    Exponential growth (``base * factor**attempt``) capped at ``cap``, then
    multiplied by a uniform jitter in ``[1 - jitter, 1 + jitter]``. Always
    returns a non-negative value bounded by ``cap * (1 + jitter)``.
    """
    raw = min(base * (factor**attempt), cap)
    r = rng if rng is not None else random
    factor_jitter = 1.0 + r.uniform(-jitter, jitter)
    return max(0.0, raw * factor_jitter)
