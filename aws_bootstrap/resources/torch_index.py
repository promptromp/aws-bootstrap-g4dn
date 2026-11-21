"""Pick the PyTorch CUDA wheel index that matches this machine's CUDA toolkit.

Uploaded to the instance and run by ``remote_setup.sh`` with the venv's Python, so
it is stdlib-only and avoids syntax newer than the instance's interpreter.

Rather than a hand-maintained list of CUDA wheel tags, which silently pinned every
launch to an old index as PyTorch published new ones, it reads the tags PyTorch
actually publishes (the ``cuNNN/`` directories of the index root), keeps those no
newer than the installed toolkit, and asks ``uv`` to resolve the packages against
each, newest first. The first index on which every package resolves to a
``+cuNNN`` build wins. Checking the resolution, not just the directory, matters:
an index can exist before it carries every package for this Python and CPU, and
the PyTorch indexes also mirror stray CPU-only wheels that would otherwise
"resolve" to the wrong build.

Usage::

    python torch_index.py 13.5 [--python ~/venv/bin/python]

Prints the winning tag (e.g. ``cu134``) on stdout, or nothing when none fits or a
lookup fails; the caller then falls back to PyPI. Each rejected tag is reported
on stderr so a downgrade is never silent.
"""

from __future__ import annotations
import argparse
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterable


INDEX_ROOT = "https://download.pytorch.org/whl/"
PACKAGES = ("torch", "torchvision")
FETCH_TIMEOUT = 30  # seconds, for the index root
RESOLVE_TIMEOUT = 300  # seconds, per `uv pip compile`

# "cu134/" -> major 13, minor 4. CUDA minors have always been one digit, so the
# last digit is the minor and the rest the major (cu100 = 10.0, cu118 = 11.8).
_TAG_LINK = re.compile(r'href="cu(\d+)(\d)/"')
_VERSION = re.compile(r"^\s*(\d+)\.(\d+)")
_PINNED = re.compile(r"^([A-Za-z0-9_.-]+)==(\S+)")

Version = tuple[int, int]


def parse_version(text: str) -> Version | None:
    """``"13.5"`` or ``"13.5.1"`` -> ``(13, 5)``; anything else -> ``None``."""
    m = _VERSION.match(text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def tag_for(version: Version) -> str:
    """``(13, 4)`` -> ``"cu134"``."""
    return f"cu{version[0]}{version[1]}"


def published_versions(index_html: str) -> list[Version]:
    """CUDA versions that have a wheel index, newest first, from the index root's links."""
    return sorted({(int(major), int(minor)) for major, minor in _TAG_LINK.findall(index_html)}, reverse=True)


def candidates(published: Iterable[Version], toolkit: Version) -> list[Version]:
    """Published versions no newer than *toolkit*, newest first."""
    return sorted((v for v in published if v <= toolkit), reverse=True)


def parse_pins(requirements: str) -> dict[str, str]:
    """``name==version`` lines (``uv pip compile`` output) -> ``{normalized name: version}``."""
    pins = {}
    for line in requirements.splitlines():
        m = _PINNED.match(line.strip())
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def cuda_builds_missing(pins: dict[str, str], tag: str, packages: Iterable[str] = PACKAGES) -> list[str]:
    """Packages that did not resolve to a ``+<tag>`` build (empty list: all did)."""
    return [p for p in packages if not pins.get(p, "").endswith(f"+{tag}")]


def fetch_index_root(url: str = INDEX_ROOT, timeout: int = FETCH_TIMEOUT) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def uv_resolve(index_url: str, python: str, extra_args: Iterable[str] = (), timeout: int = RESOLVE_TIMEOUT) -> str:
    """Resolve PACKAGES against *index_url* for *python*; returns pinned requirements text."""
    cmd = [
        "uv",
        "pip",
        "compile",
        "-",
        "--quiet",
        "--no-header",
        "--no-annotate",
        "--python",
        python,
        "--default-index",
        index_url,
        *extra_args,
    ]
    result = subprocess.run(cmd, input="\n".join(PACKAGES), capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"uv exited {result.returncode}")
    return result.stdout


def select_tag(
    toolkit: Version,
    fetch: Callable[[], str],
    resolve: Callable[[str], str],
    log: Callable[[str], None],
) -> str | None:
    """Newest published tag <= *toolkit* on which every package resolves to a CUDA build."""
    try:
        published = published_versions(fetch())
    except Exception as e:  # network, TLS, decode: fall back to PyPI, but say why
        log(f"could not read the PyTorch index ({e})")
        return None
    if not published:
        log("the PyTorch index lists no CUDA wheel indexes")
        return None
    for version in candidates(published, toolkit):
        tag = tag_for(version)
        try:
            missing = cuda_builds_missing(parse_pins(resolve(f"{INDEX_ROOT}{tag}")), tag)
        except Exception as e:  # resolution failure or timeout: try the next tag
            log(f"{tag}: not usable ({e})")
            continue
        if not missing:
            return tag
        log(f"{tag}: no {tag} build of {', '.join(missing)}")
    log(f"no PyTorch CUDA index at or below CUDA {toolkit[0]}.{toolkit[1]} serves {', '.join(PACKAGES)}")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick the PyTorch CUDA wheel index for this CUDA toolkit.")
    parser.add_argument("cuda", help="installed CUDA toolkit version, e.g. 13.5 (from nvcc)")
    parser.add_argument("--python", default=sys.executable, help="interpreter the packages are for (the venv's)")
    parser.add_argument(
        "--python-platform",
        help="resolve for another platform, e.g. x86_64-manylinux_2_28 (for checking off-instance)",
    )
    args = parser.parse_args(argv)

    def log(msg: str) -> None:
        print(f"  torch_index: {msg}", file=sys.stderr)

    toolkit = parse_version(args.cuda)
    if toolkit is None:
        log(f"unrecognised CUDA version {args.cuda!r}")
        return 0
    extra = ["--python-platform", args.python_platform] if args.python_platform else []
    tag = select_tag(
        toolkit,
        fetch=fetch_index_root,
        resolve=lambda url: uv_resolve(url, args.python, extra),
        log=log,
    )
    if tag:
        print(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
