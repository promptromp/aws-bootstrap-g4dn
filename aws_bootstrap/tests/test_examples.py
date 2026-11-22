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

_TORCH_SETUP_PY = """\
from setuptools import setup
setup(name="torch", version="{version}", packages=["torch"])
"""
_TORCH_INIT = 'class version:\n    cuda = "13.4"\n__version__ = "2.15.0+cu134"\n'
_FANCY_SETUP_PY = """\
import torch  # like a CUDA extension: needs the venv's torch at build time (--no-build-isolation)
from setuptools import setup
setup(name="Fancy_Ext", version="{version}", py_modules=["fancy_ext"], install_requires={requires!r})
"""
# Fancy_Ext 3.0 needs another torch: installing it must fail rather than replace the venv's.
_FANCY_REQUIRES = {"1.0": [], "1.2": [], "2.0": [], "3.0": ["torch==9.9"]}
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
    """A venv with pip, setuptools, the build tools and a stand-in torch 2.15.0+cu134, plus:

    - ``sdists/`` (pip's find-links): Fancy_Ext 1.0, 1.2, 2.0 and 3.0 sdists, and a prebuilt 2.0 wheel;
    - ``wheels/`` (uv's find-links): the torch wheels, including the 9.9 that Fancy_Ext 3.0 asks for.
    """
    uv = shutil.which("uv")
    if not uv or not shutil.which("bash"):
        pytest.skip("needs uv and bash")
    root = tmp_path_factory.mktemp("toolchain")
    python = root / "venv" / "bin" / "python"
    try:
        _run_quiet(uv, "venv", "--python", sys.executable, root / "venv")
        _run_quiet(uv, "pip", "install", "--python", python, "pip", "setuptools", "wheel", "ninja", "packaging")
    except subprocess.CalledProcessError as e:  # e.g. offline with a cold uv cache
        pytest.skip(f"could not build the scratch venv: {e.stderr.decode()[-300:]}")
    wheels, sdists = root / "wheels", root / "sdists"
    build_wheel = (python, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation")
    for version in ("2.15.0+cu134", "9.9"):  # real distributions, so uv knows torch's version
        src = root / "torch-src" / version
        (src / "torch").mkdir(parents=True)
        (src / "torch" / "__init__.py").write_text(_TORCH_INIT)
        (src / "setup.py").write_text(_TORCH_SETUP_PY.format(version=version))
        _run_quiet(*build_wheel, "-w", wheels, src)
    _run_quiet(uv, "pip", "install", "--python", python, "--no-index", "--find-links", wheels, "torch==2.15.0+cu134")
    for version, requires in _FANCY_REQUIRES.items():
        src = root / "src" / f"Fancy_Ext-{version}"
        src.mkdir(parents=True)
        (src / "setup.py").write_text(_FANCY_SETUP_PY.format(version=version, requires=requires))
        (src / "fancy_ext.py").write_text("")
        _run_quiet(python, "setup.py", "-q", "sdist", "-d", sdists, cwd=src)
    # A prebuilt wheel on the index: the script must still compile from source (--no-binary).
    _run_quiet(*build_wheel, "-w", sdists, root / "src" / "Fancy_Ext-2.0")
    return {"uv": uv, "python": python, "root": root, "sdists": sdists, "wheels": wheels}


def _runner(home: Path, toolchain: dict, fake_s3: Path):
    """An instance with its own HOME (so its own local cache) that uses the bucket at *fake_s3*.

    Calling it runs the script as documented (``bash -s -- PKG... < install_compiled.sh``) and
    returns ``(exit code, combined output, local cache dir or None)``.
    """
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    for tool, body in _STUBS.items():
        (bin_dir / tool).write_text(f"#!/usr/bin/env bash\n{body}")
        (bin_dir / tool).chmod(0o755)
    (bin_dir / "uv").symlink_to(toolchain["uv"])

    def run(
        *packages: str, cache: bool = True, aws_fail: bool = False, cwd: Path | None = None
    ) -> tuple[int, str, Path | None]:
        env = {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
            "VENV": str(toolchain["python"].parent.parent),
            "FAKE_S3": str(fake_s3),
            # Everything resolves locally, never from the network.
            "PIP_NO_INDEX": "1",
            "PIP_FIND_LINKS": str(toolchain["sdists"]),
            "UV_NO_INDEX": "1",
            "UV_FIND_LINKS": str(toolchain["wheels"]),
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(home.parent / "uv-cache")),
            **({"WHEEL_CACHE": "s3://bucket/wheels"} if cache else {}),
            **({"AWS_FAIL": "1"} if aws_fail else {}),
        }
        with INSTALL_COMPILED.open() as script:
            proc = subprocess.run(
                ["bash", "-s", "--", *packages],
                stdin=script,
                env=env,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=180,
            )
        caches = list((home / "wheels").glob("*"))
        return proc.returncode, proc.stdout + proc.stderr, caches[0] if caches else None

    return run


@pytest.fixture(scope="module")
def seeded_bucket(tmp_path_factory, toolchain):
    """A bucket already holding the fancy-ext 1.0 wheel, built once for the module; tests get a copy."""
    root = tmp_path_factory.mktemp("seeded")
    rc, out, _ = _runner(root / "home", toolchain, root / "s3")("fancy-ext==1.0")
    assert rc == 0, out
    return root / "s3"


@pytest.fixture
def instance(tmp_path, toolchain, request):
    """``instance()`` makes a fresh instance; all of a test's instances share one bucket.

    ``instance(seeded=True)`` starts that bucket with the fancy-ext 1.0 wheel already in it.
    """
    fake_s3 = tmp_path / "s3"
    homes = iter(range(1, 100))

    def make(*, seeded: bool = False):
        if seeded and not fake_s3.exists():
            shutil.copytree(request.getfixturevalue("seeded_bucket"), fake_s3)
        return _runner(tmp_path / f"home{next(homes)}", toolchain, fake_s3)

    make.s3 = fake_s3  # type: ignore[attr-defined]
    return make


def _installed_version(toolchain, dist: str = "fancy-ext") -> str:
    code = f"import importlib.metadata as m; print(m.version({dist!r}))"
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
    rc, out, _ = instance(seeded=True)("Fancy.Ext==1.0")  # another spelling of the same name
    assert rc == 0, out
    assert "using cached fancy_ext-1.0-py3-none-any.whl" in out
    assert "building" not in out
    assert "aws cp" not in out  # nothing new to upload
    assert _installed_version(toolchain) == "1.0"


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec,expected_version,builds",
    [
        pytest.param("fancy_ext>=1.1,<3", "2.0", True, id="cache-does-not-satisfy"),
        pytest.param("fancy-ext~=1.0", "1.0", False, id="cache-satisfies"),
        pytest.param("fancy-ext", "1.0", False, id="unpinned-reuses-cache"),
    ],
)
def test_pip_decides_whether_the_cache_satisfies_the_spec(instance, toolchain, spec, expected_version, builds):
    """The cache holds 1.0. A build must compile from source even though the index has a
    prebuilt 2.0 wheel (``--no-binary``): it was built against some other torch."""
    rc, out, _ = instance(seeded=True)(spec)
    assert rc == 0, out
    assert ("Building wheel for fancy" in out) is builds
    assert _installed_version(toolchain) == expected_version


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec,cwd",
    [
        pytest.param("{root}/src/Fancy_Ext-1.2", None, id="absolute-path"),
        pytest.param("src/Fancy_Ext-1.2", "root", id="relative-path"),
        pytest.param("fancy_ext-1.2.tar.gz", "sdists", id="sdist-file"),
        pytest.param("fancy-ext @ file://{root}/src/Fancy_Ext-1.2", None, id="url"),
    ],
)
def test_a_direct_reference_builds_but_stays_out_of_the_cache(instance, toolchain, spec, cwd):
    run = instance(seeded=True)
    rc, out, cache = run(spec.format(root=toolchain["root"]), cwd=toolchain[cwd] if cwd else None)
    assert rc == 0, out
    assert "building against torch" in out
    assert "aws cp" not in out
    assert [w.name for w in cache.iterdir()] == ["fancy_ext-1.0-py3-none-any.whl"]  # 1.2 isn't cached
    assert not (instance.s3 / "bucket" / "wheels" / cache.name / "fancy_ext-1.2-py3-none-any.whl").exists()
    assert _installed_version(toolchain) == "1.2"


@pytest.mark.slow
def test_a_wheel_that_needs_another_torch_fails_instead_of_replacing_it(instance, toolchain):
    rc, out, _ = instance()("fancy-ext==3.0")
    assert rc != 0, out
    assert _installed_version(toolchain, "torch") == "2.15.0+cu134"


@pytest.mark.slow
def test_a_spec_whose_marker_excludes_this_python_explains_itself(instance):
    rc, out, _ = instance()("fancy-ext==1.0; python_version < '3'")
    assert rc != 0
    assert "pip built no wheel (does an environment marker exclude this Python?)" in out


@pytest.mark.slow
def test_an_unreachable_cache_only_warns(instance, toolchain):
    rc, out, _ = instance()("fancy-ext==1.2", aws_fail=True)
    assert rc == 0, out
    assert "WARNING: could not read s3://bucket/wheels/" in out
    assert "WARNING: could not upload" in out
    assert _installed_version(toolchain) == "1.2"
