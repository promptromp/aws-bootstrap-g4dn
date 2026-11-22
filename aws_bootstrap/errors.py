"""The CLI's user-facing error type (no dependencies, so any module can raise it)."""

from __future__ import annotations

import click


class CLIError(click.ClickException):
    """A ClickException that displays the error message in red."""

    def show(self, file=None):  # type: ignore[no-untyped-def]
        # `file=None` + `err=True` lets click resolve stderr itself; the explicit
        # `click.get_text_stream("stderr")` it replaces is deprecated in Click 8.5.
        click.secho(f"Error: {self.format_message()}", file=file, err=True, fg="red")
