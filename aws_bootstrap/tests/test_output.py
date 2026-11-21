"""Tests for the output formatting module."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path

import click
import pytest
import yaml
from click.testing import CliRunner

from aws_bootstrap.output import OutputFormat, echo, emit, is_text


def _run(fmt: OutputFormat, body):
    """Invoke ``body(ctx)`` inside a click command whose output format is ``fmt``."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = fmt
        body(ctx)

    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0, result.output
    return result


def test_output_format_enum_values():
    assert [f.value for f in OutputFormat] == ["text", "json", "yaml", "table"]


@pytest.mark.parametrize(
    "value,serialized",
    [
        (datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC), "2025-06-15T12:30:00+00:00"),
        (Path("/home/user/.ssh/id_ed25519"), "/home/user/.ssh/id_ed25519"),
    ],
    ids=["datetime-iso", "path-str"],
)
def test_emit_json_serializes_non_json_types(value, serialized):
    result = _run(OutputFormat.JSON, lambda ctx: emit({"v": value}, ctx=ctx))
    assert json.loads(result.output) == {"v": serialized}


@pytest.mark.parametrize("fmt,load", [(OutputFormat.JSON, json.loads), (OutputFormat.YAML, yaml.safe_load)])
def test_emit_structured_round_trips(fmt, load):
    result = _run(fmt, lambda ctx: emit({"key": "value", "count": 42}, ctx=ctx))
    assert load(result.output) == {"key": "value", "count": 42}


def test_emit_table_list():
    """A list of dicts renders as a table with headers."""
    rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    result = _run(OutputFormat.TABLE, lambda ctx: emit(rows, headers={"name": "Name", "age": "Age"}, ctx=ctx))
    for text in ("Name", "Age", "Alice", "Bob"):
        assert text in result.output


def test_emit_table_dict():
    """A single dict renders as key-value pairs."""
    result = _run(OutputFormat.TABLE, lambda ctx: emit({"instance_id": "i-abc123", "state": "running"}, ctx=ctx))
    for text in ("instance_id", "i-abc123", "running"):
        assert text in result.output


@pytest.mark.parametrize("fmt,shown", [(OutputFormat.TEXT, True), (OutputFormat.JSON, False)])
def test_echo_only_in_text_mode(fmt, shown):
    result = _run(fmt, lambda ctx: echo("Hello world"))
    assert result.output == ("Hello world\n" if shown else "")


@pytest.mark.parametrize("fmt,expected", [(OutputFormat.TEXT, True), (OutputFormat.JSON, False)])
def test_is_text(fmt, expected):
    seen = []
    _run(fmt, lambda ctx: seen.append(is_text(ctx)))
    assert seen == [expected]
