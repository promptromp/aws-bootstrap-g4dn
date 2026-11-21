"""pytest-impacted's rules must cover what its import analysis cannot see.

The pre-commit hook and pull-request CI run only the tests a change can reach
(``pytest --impacted``, configured in pyproject's ``[tool.pytest.ini_options]``).
The plugin follows Python imports; a file a test *reads* (a shell script, a JSON
template, a payload it checks for) needs an ``impacted_map`` or
``impacted_invalidate_all`` rule, or a change to it would silently select nothing.
"""

from __future__ import annotations
import subprocess
import tomllib
from pathlib import Path

import pytest


strategies = pytest.importorskip("pytest_impacted.strategies")  # dev-only dependency

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not (PYPROJECT.is_file() and (ROOT / ".git").exists()),
    reason="needs the source checkout (pyproject.toml and git), not an installed package",
)


def _config() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]


def _rule_globs(cfg: dict) -> list[str]:
    return [*cfg["impacted_invalidate_all"], *(entry.split("=", 1)[0] for entry in cfg["impacted_map"])]


def _tracked(path: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", path], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout.split()


def test_every_non_python_package_file_has_a_rule():
    """A new resource (script, template, notebook) needs a rule before a change to it is missed."""
    globs = _rule_globs(_config())
    files = [f for f in _tracked("aws_bootstrap") if not f.endswith(".py")]
    assert files, "expected tracked resource files under aws_bootstrap/"
    missing = [f for f in files if not strategies.matches_any_glob(f, globs)]
    assert not missing, f"add an impacted_map / impacted_invalidate_all rule for: {missing}"


def test_dependency_and_ci_changes_invalidate_everything():
    globs = _config()["impacted_invalidate_all"]
    for path in ("pyproject.toml", "uv.lock", ".pre-commit-config.yaml", ".github/workflows/ci.yml"):
        assert strategies.matches_any_glob(path, globs), path


def test_map_targets_exist():
    for entry in _config()["impacted_map"]:
        glob, _, target = entry.partition("=")
        assert glob and target, f"malformed impacted_map entry: {entry!r}"
        assert (ROOT / target).exists(), f"impacted_map target missing: {target}"


def test_module_and_tests_dir_point_at_real_directories():
    cfg = _config()
    assert (ROOT / cfg["impacted_module"] / "__init__.py").is_file()
    assert (ROOT / cfg["impacted_tests_dir"]).is_dir()
