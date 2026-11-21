"""resources/torch_index.py: discovering the PyTorch CUDA wheel index for a toolkit.

The helper runs on the instance, so these tests drive its pure functions and
inject fakes for the two side effects (fetching the index root, resolving with uv).
"""

from __future__ import annotations
import re
import subprocess
from pathlib import Path

import pytest

from aws_bootstrap.resources import torch_index as ti
from aws_bootstrap.ssh import SETUP_PAYLOAD


RESOURCES = Path(ti.__file__).parent

# Shape of https://download.pytorch.org/whl/ — CUDA dirs mixed with others.
INDEX_HTML = """
<html><body>
<a href="cpu/">cpu</a>
<a href="cu92/">cu92</a>
<a href="cu100/">cu100</a>
<a href="cu118/">cu118</a>
<a href="cu129/">cu129</a>
<a href="cu130/">cu130</a>
<a href="cu132/">cu132</a>
<a href="cu134/">cu134</a>
<a href="cu134/">cu134</a>
<a href="rocm6.4/">rocm6.4</a>
<a href="xpu/">xpu</a>
<a href="torch/">torch</a>
</body></html>
"""


def _pins(tag: str | None, torchvision_tag: str | None = "same") -> str:
    """uv pip compile output; tag=None means a CPU build (no local version)."""
    tv = tag if torchvision_tag == "same" else torchvision_tag
    torch = f"torch==2.15.0+{tag}" if tag else "torch==2.0.1"
    vision = f"torchvision==0.29.2+{tv}" if tv else "torchvision==0.17.0"
    return f"filelock==3.20.0\n{torch}\n{vision}\ntyping-extensions==4.16.0\n"


class _Log(list):
    def __call__(self, msg: str) -> None:
        self.append(msg)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("13.5", (13, 5)),
        ("13.5.1", (13, 5)),  # patch level ignored — the old `tr -d .` made this 1351
        (" 12.9\n", (12, 9)),
        ("10.0", (10, 0)),
        ("", None),
        ("N/A", None),
        ("garbage", None),
    ],
)
def test_parse_version(text, expected):
    assert ti.parse_version(text) == expected


@pytest.mark.parametrize("version,tag", [((13, 4), "cu134"), ((10, 0), "cu100"), ((9, 2), "cu92")])
def test_tag_for(version, tag):
    assert ti.tag_for(version) == tag


def test_published_versions_reads_cuda_dirs_only_newest_first():
    assert ti.published_versions(INDEX_HTML) == [
        (13, 4),
        (13, 2),
        (13, 0),
        (12, 9),
        (11, 8),
        (10, 0),
        (9, 2),
    ]


def test_published_versions_empty_page():
    assert ti.published_versions("<html></html>") == []


def test_candidates_are_capped_at_the_toolkit_and_compare_numerically():
    published = ti.published_versions(INDEX_HTML)
    # 13.1: no cu131 exists, so the newest usable index is cu130 — never cu132.
    assert ti.candidates(published, (13, 1))[:2] == [(13, 0), (12, 9)]
    # Numeric, not string, ordering: cu92 (9.2) sorts below cu118 (11.8).
    assert ti.candidates(published, (11, 8)) == [(11, 8), (10, 0), (9, 2)]
    assert ti.candidates(published, (8, 0)) == []


def test_parse_pins_normalizes_names_and_skips_comments():
    text = "# via torch\nTorch==2.15.0+cu134\n    # via -r -\ntyping_extensions==4.16.0\n\n"
    assert ti.parse_pins(text) == {"torch": "2.15.0+cu134", "typing-extensions": "4.16.0"}


def test_cuda_builds_missing():
    assert ti.cuda_builds_missing(ti.parse_pins(_pins("cu134")), "cu134") == []
    assert ti.cuda_builds_missing(ti.parse_pins(_pins("cu134", None)), "cu134") == ["torchvision"]
    assert ti.cuda_builds_missing({}, "cu134") == ["torch", "torchvision"]


# ---------------------------------------------------------------------------
# select_tag
# ---------------------------------------------------------------------------


def _resolver(results: dict[str, str | Exception]):
    """Fake uv resolution keyed by tag; records which tags were tried."""
    tried: list[str] = []

    def resolve(url: str) -> str:
        tag = url.rstrip("/").rsplit("/", 1)[-1]
        tried.append(tag)
        outcome = results.get(tag, RuntimeError("no solution"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return resolve, tried


def test_select_tag_picks_newest_index_at_or_below_the_toolkit():
    resolve, tried = _resolver({"cu134": _pins("cu134"), "cu132": _pins("cu132")})
    log = _Log()
    assert ti.select_tag((13, 5), lambda: INDEX_HTML, resolve, log) == "cu134"
    assert tried == ["cu134"]
    assert log == []


def test_select_tag_never_tries_an_index_newer_than_the_toolkit():
    resolve, tried = _resolver({"cu132": _pins("cu132"), "cu130": _pins("cu130")})
    assert ti.select_tag((13, 2), lambda: INDEX_HTML, resolve, _Log()) == "cu132"
    assert "cu134" not in tried


def test_select_tag_skips_an_index_missing_a_cuda_build_and_says_why():
    """An index can exist before it serves every package — or only via a stray CPU wheel."""
    resolve, tried = _resolver({"cu134": _pins("cu134", None), "cu132": _pins("cu132")})
    log = _Log()
    assert ti.select_tag((13, 5), lambda: INDEX_HTML, resolve, log) == "cu132"
    assert tried == ["cu134", "cu132"]
    assert any("cu134" in m and "torchvision" in m for m in log)


def test_select_tag_skips_an_index_that_fails_to_resolve():
    resolve, _ = _resolver({"cu134": RuntimeError("No solution found"), "cu132": _pins("cu132")})
    log = _Log()
    assert ti.select_tag((13, 5), lambda: INDEX_HTML, resolve, log) == "cu132"
    assert any("cu134" in m and "No solution found" in m for m in log)


def test_select_tag_falls_back_when_the_index_cannot_be_read():
    def fetch():
        raise OSError("network unreachable")

    resolve, tried = _resolver({})
    log = _Log()
    assert ti.select_tag((13, 5), fetch, resolve, log) is None
    assert tried == []
    assert any("network unreachable" in m for m in log)


def test_select_tag_none_when_nothing_fits():
    resolve, _ = _resolver({})
    log = _Log()
    assert ti.select_tag((13, 5), lambda: INDEX_HTML, resolve, log) is None
    assert any("no PyTorch CUDA index" in m for m in log)


def test_select_tag_none_when_index_lists_no_cuda():
    log = _Log()
    assert ti.select_tag((13, 5), lambda: "<a href='cpu/'>cpu</a>", _resolver({})[0], log) is None
    assert log


# ---------------------------------------------------------------------------
# side-effect wrappers and the CLI entry point
# ---------------------------------------------------------------------------


def test_uv_resolve_builds_the_compile_command(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout=_pins("cu134"), stderr="")

    monkeypatch.setattr(ti.subprocess, "run", fake_run)
    out = ti.uv_resolve("https://download.pytorch.org/whl/cu134", "/venv/bin/python", ["--python-platform", "x"])
    assert "torch==2.15.0+cu134" in out
    cmd = seen["cmd"]
    assert cmd[:3] == ["uv", "pip", "compile"]
    assert cmd[cmd.index("--default-index") + 1] == "https://download.pytorch.org/whl/cu134"
    assert cmd[cmd.index("--python") + 1] == "/venv/bin/python"
    assert cmd[-2:] == ["--python-platform", "x"]
    assert seen["kwargs"]["input"] == "torch\ntorchvision"
    assert seen["kwargs"]["timeout"] == ti.RESOLVE_TIMEOUT  # never unbounded: remote_setup.sh is


def test_uv_resolve_raises_with_uvs_last_error_line(monkeypatch):
    monkeypatch.setattr(
        ti.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error\n  No solution found\n"),
    )
    with pytest.raises(RuntimeError, match="No solution found"):
        ti.uv_resolve("u", "p")


def test_main_prints_only_the_tag_on_stdout(monkeypatch, capsys):
    monkeypatch.setattr(ti, "fetch_index_root", lambda: INDEX_HTML)
    monkeypatch.setattr(ti, "uv_resolve", lambda url, python, extra: _pins(url.rsplit("/", 1)[-1]))
    assert ti.main(["13.5", "--python", "/venv/bin/python"]) == 0
    assert capsys.readouterr().out == "cu134\n"


def test_main_unrecognised_version_prints_nothing(capsys):
    assert ti.main(["N/A"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unrecognised" in captured.err


# ---------------------------------------------------------------------------
# wiring: what remote_setup.sh reads from /tmp must be uploaded
# ---------------------------------------------------------------------------


def test_every_tmp_file_remote_setup_reads_is_in_the_setup_payload():
    script = (RESOURCES / "remote_setup.sh").read_text()
    referenced = set(re.findall(r"/tmp/([A-Za-z0-9_.-]+)", script))
    assert "torch_index.py" in referenced
    assert referenced <= set(SETUP_PAYLOAD)


def test_every_setup_payload_file_ships_in_the_package():
    for name in SETUP_PAYLOAD:
        assert (RESOURCES / name).is_file(), name


def test_remote_setup_has_no_hardcoded_cuda_tag_list():
    assert "KNOWN_CUDA_TAGS" not in (RESOURCES / "remote_setup.sh").read_text()
