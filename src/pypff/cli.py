from __future__ import annotations

from typing import Annotated, Any

import typer

# Local Imports
from .util.cli import BaseLazyGroup, display_tree_callback


class PypffLazyGroup(BaseLazyGroup):
    """
    Custom Click Group that lazy-loads commands from other modules.
    Ensures that heavy dependencies (like NumPy or Rich) aren't loaded
    until a specific command is actually executed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        lazy_mapping = {
            "show": ("pypff._cli.root", "show", "Explore the structure of a PanoSETI run."),
            "test": ("pypff._cli.test", "app", "Run pypff test suite."),
            "profile": ("pypff._cli.profile", "app", "Run performance benchmarks on a PanoSETI run."),
            "zarr": ("pypff._cli.zarr", "app", "PFF → Zarr v3 conversion tools."),
        }
        super().__init__(*args, lazy_mapping=lazy_mapping, **kwargs)


app = typer.Typer(
    cls=PypffLazyGroup,
    help="Pypff Management CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    tree: Annotated[
        bool,
        typer.Option(
            "--tree",
            "-t",
            help="Display the command tree and exit.",
            callback=display_tree_callback,
        ),
    ] = False,
) -> None:
    """PYPFF I/O CLI."""
    if ctx.invoked_subcommand is None and not tree:
        print(ctx.get_help())


if __name__ == "__main__":
    app()
