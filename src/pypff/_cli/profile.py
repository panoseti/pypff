from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pypff.profiling import Profiler

app = typer.Typer()

@app.command()
def run(
    run_dir: Annotated[Path, typer.Argument(help="Path to the .pffd run directory.")],
    products: Annotated[list[str] | None, typer.Option("--product", "-p", help="Specific products to test. Defaults to all.")] = None,
    n_frames: Annotated[int, typer.Option("--frames", "-n", help="Number of frames to test per product.")] = 1000,
    step: Annotated[int, typer.Option("--step", "-s", help="Step size for strided read test.")] = 10,
    random_samples: Annotated[int, typer.Option("--random", "-r", help="Number of random access samples.")] = 100,
) -> None:
    """Run performance benchmarks on a PanoSETI run."""
    console = Console()
    
    if not run_dir.exists():
        console.print(f"[red]❌ Run directory {run_dir} does not exist.[/]")
        raise typer.Exit(1)
    
    profiler = Profiler(run_dir)
    target_products = products or profiler.run.list_products()
    
    if not target_products:
        console.print("[yellow]⚠️ No data products found in the directory.[/]")
        return

    with console.status("[bold green]Running benchmarks...[/]"):
        for prod in target_products:
            try:
                # 1. Sequential Read
                profiler.profile_sequential_read(prod, n_frames=n_frames)
                
                # 2. Bulk Read
                profiler.profile_bulk_read(prod, n_frames=n_frames)
                
                # 3. Strided Read
                profiler.profile_strided_read(prod, step=step, n_frames=n_frames // step)
                
                # 4. Random Access
                profiler.profile_random_access(prod, n_samples=random_samples)
                
                # 5. Metadata
                profiler.profile_metadata_extraction(prod)
            except Exception as e:
                console.print(f"[red]Error profiling {prod}: {e}[/]")

    profiler.display_results()

if __name__ == "__main__":
    app()
