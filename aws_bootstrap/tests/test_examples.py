"""The runnable examples under examples/.

Every shell example must parse. install_compiled.sh is run for real: real pip and
uv in a scratch venv, building tiny local sdists whose setup.py imports a stand-in
``torch``, as a CUDA extension's does. Only what an instance provides is faked:
``nvidia-smi``, ``nvcc``, and ``aws s3`` (backed by a local directory).
"""

from __future__ import annotations
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
INSTALL_COMPILED = EXAMPLES / "compiled-extensions" / "install_compiled.sh"


@pytest.mark.parametrize("script", sorted(EXAMPLES.rglob("*.sh")), ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_example_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_install_compiled_without_packages_is_a_usage_error(tmp_path):
    proc = subprocess.run(["bash", str(INSTALL_COMPILED)], capture_output=True, text=True, env={"HOME": str(tmp_path)})
    assert proc.returncode == 2
    assert "usage:" in proc.stderr


# ---------------------------------------------------------------------------
# install_compiled.sh, end to end (slow: builds a venv and real wheels)
# ---------------------------------------------------------------------------

_FAKE_TORCH = 'class version:\n    cuda = "13.4"\n__version__ = "2.15.0+cu134"\n'
_SETUP_PY = """\
import torch  # like a CUDA extension: needs the venv's torch at build time (--no-build-isolation)
from setuptools import setup
setup(name="Fancy_Ext", version="{version}", py_modules=["fancy_ext"])
"""
_STUBS = {
    "nvidia-smi": "echo 8.9",
    "nvcc": 'echo "Cuda compilation tools, release 13.5, V13.5.1"',
    # aws s3 sync|cp SRC DST, with s3:// mapped onto $FAKE_S3. Like the real one, syncing
    # from an absent prefix succeeds; $AWS_FAIL=1 makes every call fail.
    "aws": r"""
[ -n "${AWS_FAIL:-}" ] && { echo "aws: AccessDenied" >&2; exit 1; }
src=${3/s3:\/\//$FAKE_S3/}; dst=${4/s3:\/\//$FAKE_S3/}
echo "aws $2 $3" >&2
mkdir -p "$dst"
if [ "$2" = sync ]; then [ -d "$src" ] && cp -R "$src". "$dst"; exit 0; fi
cp "$src" "$dst"
""",
}
KEY_PREFIX = f"torch2.15.0_cu134-cuda13.4-nvcc13.5-cp{sys.version_info[0]}{sys.version_info[1]}-{platform.machine()}-"


def _run_quiet(*cmd: str | Path, **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True, **kw)


@pytest.fixture(scope="module")
def toolchain(tmp_path_factory):
    """A venv with pip, setuptools, the build tools and a stand-in torch, plus Fancy_Ext 1.0/1.2/2.0 sdists."""
    uv = shutil.which("uv")
    if not uv or not shutil.which("bash"):
        pytest.skip("needs uv and bash")
    root = tmp_path_factory.mktemp("toolchain")
    venv = root / "venv"
    python = venv / "bin" / "python"
    try:
        _run_quiet(uv, "venv", "--python", sys.executable, venv)
        _run_quiet(uv, "pip", "install", "--python", python, "pip", "setuptools", "wheel", "ninja", "packaging")
    except subprocess.CalledProcessError as e:  # e.g. offline with a cold uv cache
        pytest.skip(f"could not build the scratch venv: {e.stderr.decode()[-300:]}")
    purelib = subprocess.check_output([python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"])
    torch = Path(purelib.decode().strip()) / "torch"
    torch.mkdir()
    (torch / "__init__.py").write_text(_FAKE_TORCH)
    sdists = root / "sdists"
    for version in ("1.0", "1.2", "2.0"):
        src = root / "src" / f"Fancy_Ext-{version}"
        src.mkdir(parents=True)
        (src / "setup.py").write_text(_SETUP_PY.format(version=version))
        (src / "fancy_ext.py").write_text("")
        _run_quiet(python, "setup.py", "-q", "sdist", "-d", sdists, cwd=src)
    return {"uv": uv, "python": python, "sdists": sdists, "src": root / "src"}


@pytest.fixture
def instance(tmp_path, toolchain):
    """``instance()`` makes a fresh instance (its own HOME, so its own local cache); all share one fake bucket.

    Its runner runs the script as documented (``bash -s -- PKG... < install_compiled.sh``) and
    returns ``(exit code, combined output, local cache dir or None)``.
    """
    fake_s3 = tmp_path / "s3"
    count = 0

    def make():
        nonlocal count
        count += 1
        home = tmp_path / f"home{count}"
        bin_dir = home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        for tool, body in _STUBS.items():
            (bin_dir / tool).write_text(f"#!/usr/bin/env bash\n{body}")
            (bin_dir / tool).chmod(0o755)
        (bin_dir / "uv").symlink_to(toolchain["uv"])

        def run(*packages: str, cache: bool = True, aws_fail: bool = False) -> tuple[int, str, Path | None]:
            env = {
                "HOME": str(home),
                "PATH": "/usr/bin:/bin",
                "VENV": str(toolchain["python"].parent.parent),
                "FAKE_S3": str(fake_s3),
                # Builds resolve from the local sdists, never the network.
                "PIP_NO_INDEX": "1",
                "PIP_FIND_LINKS": str(toolchain["sdists"]),
                "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(tmp_path / "uv-cache")),
                **({"WHEEL_CACHE": "s3://bucket/wheels"} if cache else {}),
                **({"AWS_FAIL": "1"} if aws_fail else {}),
            }
            with INSTALL_COMPILED.open() as script:
                proc = subprocess.run(
                    ["bash", "-s", "--", *packages], stdin=script, env=env, capture_output=True, text=True, timeout=180
                )
            caches = list((home / "wheels").glob("*"))
            return proc.returncode, proc.stdout + proc.stderr, caches[0] if caches else None

        return run

    make.s3 = fake_s3  # type: ignore[attr-defined]
    return make


def _installed_version(toolchain) -> str:
    code = "import importlib.metadata as m; print(m.version('fancy-ext'))"
    return subprocess.check_output([toolchain["python"], "-c", code]).decode().strip()


@pytest.mark.slow
def test_builds_against_the_venv_torch_uploads_then_installs(instance, toolchain):
    rc, out, cache = instance()("fancy-ext==1.0")
    assert rc == 0, out
    assert cache is not None and cache.name.startswith(KEY_PREFIX) and cache.name.endswith("-sm89")
    assert "fancy-ext==1.0: building against torch 2.15.0+cu134" in out
    assert (cache / "fancy_ext-1.0-py3-none-any.whl").exists()
    assert (instance.s3 / "bucket" / "wheels" / cache.name / "fancy_ext-1.0-py3-none-any.whl").exists()
    assert out.index("aws cp") < out.index("Installed")  # uploaded as soon as it was built
    assert _installed_version(toolchain) == "1.0"


@pytest.mark.slow
def test_the_next_instance_installs_from_the_s3_cache(instance, toolchain):
    instance()("fancy-ext==1.0")
    rc, out, _ = instance()("Fancy.Ext==1.0")  # another spelling of the same name
    assert rc == 0, out
    assert "using cached fancy_ext-1.0-py3-none-any.whl" in out
    assert "building" not in out
    assert "aws cp" not in out  # nothing new to upload
    assert _installed_version(toolchain) == "1.0"


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec,expected_version,builds",
    [
        pytest.param("fancy_ext>=1.1", "2.0", True, id="cache-does-not-satisfy"),
        pytest.param("fancy-ext~=1.0", "1.0", False, id="cache-satisfies"),
        pytest.param("fancy-ext", "1.0", False, id="unpinned-reuses-cache"),
    ],
)
def test_pip_decides_whether_the_cache_satisfies_the_spec(instance, toolchain, spec, expected_version, builds):
    run = instance()
    run("fancy-ext==1.0")
    rc, out, _ = run(spec)
    assert rc == 0, out
    assert ("building" in out) is builds
    assert _installed_version(toolchain) == expected_version


@pytest.mark.slow
def test_a_direct_reference_builds_but_stays_out_of_the_cache(instance, toolchain):
    run = instance()
    run("fancy-ext==1.0")
    rc, out, cache = run(str(toolchain["src"] / "Fancy_Ext-1.2"))
    assert rc == 0, out
    assert "building against torch" in out
    assert "aws cp" not in out
    assert [w.name for w in cache.iterdir()] == ["fancy_ext-1.0-py3-none-any.whl"]  # 1.2 isn't cached
    assert _installed_version(toolchain) == "1.2"


@pytest.mark.slow
def test_an_unreachable_cache_only_warns(instance, toolchain):
    rc, out, _ = instance()("fancy-ext==1.2", aws_fail=True)
    assert rc == 0, out
    assert "WARNING: could not read s3://bucket/wheels/" in out
    assert "WARNING: could not upload" in out
    assert _installed_version(toolchain) == "1.2"
