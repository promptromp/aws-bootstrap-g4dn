"""Tests for the output formatting module."""

from __future__ import annotations
import json
from datetime import UTC, datetime
from pathlib import Path

import click
import yaml
from click.testing import CliRunner

from aws_bootstrap.output import OutputFormat, echo, emit, is_text


def test_output_format_enum_values():
    assert OutputFormat.TEXT.value == "text"
    assert OutputFormat.JSON.value == "json"
    assert OutputFormat.YAML.value == "yaml"
    assert OutputFormat.TABLE.value == "table"


def test_serialize_datetime():
    """datetime objects should serialize to ISO format strings."""
    dt = datetime(2025, 6, 15, 12, 30, 0, tzinfo=UTC)

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.JSON
        emit({"timestamp": dt}, ctx=ctx)

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["timestamp"] == "2025-06-15T12:30:00+00:00"


def test_serialize_path():
    """Path objects should serialize to strings."""
    p = Path("/home/user/.ssh/id_ed25519")

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.JSON
        emit({"path": p}, ctx=ctx)

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["path"] == "/home/user/.ssh/id_ed25519"


def test_emit_json():
    """emit() should produce valid JSON in JSON mode."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.JSON
        emit({"key": "value", "count": 42}, ctx=ctx)

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data == {"key": "value", "count": 42}


def test_emit_yaml():
    """emit() should produce valid YAML in YAML mode."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.YAML
        emit({"key": "value", "count": 42}, ctx=ctx)

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    data = yaml.safe_load(result.output)
    assert data == {"key": "value", "count": 42}


def test_emit_table_list():
    """emit() should render a list of dicts as a table with headers."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.TABLE
        emit(
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
            headers={"name": "Name", "age": "Age"},
            ctx=ctx,
        )

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "Name" in result.output
    assert "Age" in result.output
    assert "Alice" in result.output
    assert "Bob" in result.output


def test_emit_table_dict():
    """emit() should render a single dict as key-value pairs."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.TABLE
        emit({"instance_id": "i-abc123", "state": "running"}, ctx=ctx)

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "instance_id" in result.output
    assert "i-abc123" in result.output
    assert "running" in result.output


def test_echo_suppressed_in_json_mode():
    """echo() should produce no output when format is JSON."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.JSON
        echo("This should not appear")

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert result.output == ""


def test_echo_emits_in_text_mode():
    """echo() should work normally in text mode."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.TEXT
        echo("Hello world")

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert "Hello world" in result.output


def test_is_text_default():
    """is_text() should return True when no context is set (default behavior)."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.TEXT
        assert is_text(ctx) is True

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0


def test_is_text_false_for_json():
    """is_text() should return False when format is JSON."""

    @click.command()
    @click.pass_context
    def cli(ctx):
        ctx.ensure_object(dict)
        ctx.obj["output_format"] = OutputFormat.JSON
        assert is_text(ctx) is False

    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0
