"""Output formatting for structured CLI output (JSON, YAML, table, text)."""

from __future__ import annotations
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import click


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    TABLE = "table"


def get_format(ctx: click.Context | None = None) -> OutputFormat:
    """Return the current output format from the click context."""
    if ctx is None:
        ctx = click.get_current_context(silent=True)
    if ctx is None or ctx.obj is None:
        return OutputFormat.TEXT
    return ctx.obj.get("output_format", OutputFormat.TEXT)


def is_text(ctx: click.Context | None = None) -> bool:
    """Return True if the current output format is text (default)."""
    return get_format(ctx) == OutputFormat.TEXT


def _default_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def emit(data: dict | list, *, headers: dict[str, str] | None = None, ctx: click.Context | None = None) -> None:
    """Emit structured data in the configured output format.

    For JSON/YAML: serializes the data directly.
    For TABLE: renders using tabulate. If *data* is a list of dicts, uses
    *headers* mapping ``{dict_key: column_label}`` for column selection/ordering.
    If *data* is a single dict, renders as key-value pairs.
    """
    fmt = get_format(ctx)

    if fmt == OutputFormat.JSON:
        click.echo(json.dumps(data, indent=2, default=_default_serializer))
        return

    if fmt == OutputFormat.YAML:
        import yaml  # noqa: PLC0415

        # Convert datetime/Path objects before YAML dump
        prepared = json.loads(json.dumps(data, default=_default_serializer))
        click.echo(yaml.dump(prepared, default_flow_style=False, sort_keys=False).rstrip())
        return

    if fmt == OutputFormat.TABLE:
        from tabulate import tabulate  # noqa: PLC0415

        table_data = data
        # Unwrap dict-wrapped lists (e.g. {"instances": [...]}) for table rendering
        if isinstance(data, dict) and headers:
            for v in data.values():
                if isinstance(v, list):
                    table_data = v
                    break

        if isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
            if headers:
                keys = list(headers.keys())
                col_labels = list(headers.values())
                rows = [[row.get(k, "") for k in keys] for row in table_data]
            else:
                col_labels = list(table_data[0].keys())
                keys = col_labels
                rows = [[row.get(k, "") for k in keys] for row in table_data]
            click.echo(tabulate(rows, headers=col_labels, tablefmt="simple"))
        elif isinstance(table_data, dict):
            rows = [[k, v] for k, v in table_data.items()]
            click.echo(tabulate(rows, headers=["Key", "Value"], tablefmt="simple"))
        elif isinstance(table_data, list):
            # Empty list
            click.echo("(no data)")
        return

    # TEXT format: emit() is a no-op in text mode (text output is handled inline)


def echo(msg: str = "", **kwargs: Any) -> None:
    """Wrap ``click.echo``; silent in non-text output modes."""
    if is_text():
        click.echo(msg, **kwargs)


def secho(msg: str = "", **kwargs: Any) -> None:
    """Wrap ``click.secho``; silent in non-text output modes."""
    if is_text():
        click.secho(msg, **kwargs)
