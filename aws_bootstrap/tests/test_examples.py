"""The runnable examples under examples/: shell syntax, and install_compiled.sh's wheel cache with stub tools."""

from __future__ import annotations
import platform
import shutil
import subprocess
from pathlib import Path

import pytest


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
INSTALL_COMPILED = EXAMPLES / "compiled-extensions" / "install_compiled.sh"
KEY = f"torch2.15.0_cu134-cp312-{platform.machine()}-sm89"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytest.mark.parametrize("script", sorted(EXAMPLES.rglob("*.sh")), ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_example_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


# Stand-ins for what the instance provides. Each logs its call to $STUB_LOG.
# `python -m pip wheel` "builds" <name>-<version>-cp312-cp312-linux.whl
# (version 9.9 when unpinned); `aws s3 sync` copies between s3:// and $FAKE_S3.
_STUBS = {
    "python": r"""
case "$1 $2" in
  "-c import torch"*) echo "2.15.0+cu134" ;;
  "-c import sys"*) echo "cp312" ;;
  "-m pip")
    for a in "$@"; do case "$prev" in --wheel-dir) dir=$a ;; esac; prev=$a; spec=$a; done
    name=${spec%%==*}; version=9.9; [[ "$spec" == *==* ]] && version=${spec#*==}
    echo "build $spec" >> "$STUB_LOG"
    touch "$dir/$(printf '%s' "$name" | tr -s -- '-_.' '_')-$version-cp312-cp312-linux.whl" ;;
esac
""",
    # Logs installs of built wheels only, not the build tools (that call ends with "packaging").
    "uv": r"""
last=${*: -1}
[ "$1 $2" = "pip install" ] && [ "$last" != packaging ] && echo "install $last" >> "$STUB_LOG"
true
""",
    "nvidia-smi": "echo 8.9",
    "aws": r"""
src=${3/s3:\/\//$FAKE_S3/}; dst=${4/s3:\/\//$FAKE_S3/}
echo "sync $3 $4" >> "$STUB_LOG"
[ -d "$src" ] || exit 1
mkdir -p "$dst" && cp -R "$src". "$dst"
""",
}


@pytest.fixture
def instance(tmp_path):
    """A fresh instance (its own HOME and ~/venv); call it again for another one. All share one fake S3."""
    fake_s3 = tmp_path / "s3"
    count = 0

    def make():
        nonlocal count
        count += 1
        home = tmp_path / f"home{count}"
        (home / "venv" / "bin").mkdir(parents=True)
        (home / ".local" / "bin").mkdir(parents=True)
        for tool, body in _STUBS.items():
            path = home / ("venv/bin" if tool == "python" else ".local/bin") / tool
            path.write_text(f"#!/usr/bin/env bash\n{body}")
            path.chmod(0o755)
        log = home / "calls.log"

        def run(*packages: str, cache: bool = False) -> tuple[subprocess.CompletedProcess, list[str]]:
            env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "STUB_LOG": str(log), "FAKE_S3": str(fake_s3)}
            if cache:
                env["WHEEL_CACHE"] = "s3://bucket/wheels"
            result = subprocess.run(["bash", str(INSTALL_COMPILED), *packages], env=env, capture_output=True, text=True)
            calls = log.read_text().splitlines() if log.exists() else []
            log.unlink(missing_ok=True)
            return result, calls

        run.home = home  # type: ignore[attr-defined]
        return run

    make.s3 = fake_s3  # type: ignore[attr-defined]
    return make


def test_builds_once_then_installs_the_cached_wheel(instance):
    run = instance()
    first, calls = run("causal-conv1d==1.5.0")
    assert first.returncode == 0, first.stderr
    wheel = run.home / "wheels" / KEY / "causal_conv1d-1.5.0-cp312-cp312-linux.whl"
    assert calls == ["build causal-conv1d==1.5.0", f"install {wheel}"]

    again, calls = run("causal-conv1d==1.5.0")
    assert again.returncode == 0, again.stderr
    assert calls == [f"install {wheel}"]  # no rebuild
    assert "using cached causal_conv1d-1.5.0" in again.stdout


def test_a_different_pinned_version_is_built_not_reused(instance):
    run = instance()
    run("causal-conv1d==1.5.0")
    _, calls = run("causal-conv1d==1.6.0")
    assert calls[0] == "build causal-conv1d==1.6.0"


def test_an_unpinned_name_reuses_the_newest_cached_version(instance):
    run = instance()
    run("causal-conv1d==1.5.0")
    run("causal-conv1d==1.10.0")
    _, calls = run("causal-conv1d")
    assert calls == [f"install {run.home / 'wheels' / KEY / 'causal_conv1d-1.10.0-cp312-cp312-linux.whl'}"]


def test_the_s3_cache_spares_the_next_instance_the_build(instance):
    first, calls = instance()("causal-conv1d==1.5.0", cache=True)
    assert first.returncode == 0, first.stderr
    remote = f"s3://bucket/wheels/{KEY}/"
    assert calls[0].startswith(f"sync {remote} ")  # tried the cache first
    assert "build causal-conv1d==1.5.0" in calls
    assert calls[-1].startswith("sync ") and calls[-1].endswith(f" {remote}")  # uploaded the new wheel
    assert (instance.s3 / "bucket" / "wheels" / KEY / "causal_conv1d-1.5.0-cp312-cp312-linux.whl").exists()

    second, calls = instance()("causal-conv1d==1.5.0", cache=True)
    assert second.returncode == 0, second.stderr
    assert not any(c.startswith("build") for c in calls)
    assert not any(c.endswith(remote) for c in calls)  # nothing new to upload


def test_an_unreadable_cache_only_warns(instance):
    result, calls = instance()("causal-conv1d==1.5.0", cache=True)  # the fake bucket is still empty
    assert result.returncode == 0
    assert "could not read s3://bucket/wheels" in result.stderr
    assert "build causal-conv1d==1.5.0" in calls


def test_no_packages_is_a_usage_error(instance):
    result, _ = instance()()
    assert result.returncode == 2
    assert "usage:" in result.stderr
