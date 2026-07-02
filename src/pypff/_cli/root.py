from pathlib import Path
from typing import Annotated

import typer

from pypff import PanosetiRun

app = typer.Typer()

@app.command()
def show(
    run_dir: Annotated[Path, typer.Argument(help="Path to the .pffd run directory.")],
    details: Annotated[bool, typer.Option("--details", "-d", help="Show individual PFF files.")] = False,
) -> None:
    """Explore the structure of a PanoSETI run."""
    if not run_dir.exists():
        print(f"❌ Run directory {run_dir} does not exist.")
        raise typer.Exit(1)
    
    run = PanosetiRun(run_dir)
    run.show(details=details)
