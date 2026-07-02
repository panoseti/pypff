from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="PFF → Zarr v3 conversion tools.")


@app.command()
def convert(
    obs_dir: Annotated[Path, typer.Argument(help="Input .pffd observation directory")],
    out_dir: Annotated[Path, typer.Argument(help="Output directory for .zarr stores")],
    codec: Annotated[str, typer.Option(help="Compression codec: zstd, blosc-lz4, gzip, none")] = "zstd",
    level: Annotated[int, typer.Option(help="Compression level (codec-specific)")] = 3,
    time_chunk: Annotated[
        int, typer.Option(help="Frames per time chunk (0 = auto-size to ~8 MB)")
    ] = 0,
    shard_factor: Annotated[
        int, typer.Option(
            "--shard-factor",
            help=(
                "Inner chunks per shard file (0 = no sharding). "
                "Recommended: 16 for BeeGFS/HPC to reduce file count ~78x."
            )
        )
    ] = 0,
) -> None:
    """Convert all data products in a .pffd observation directory to Zarr v3 stores."""
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    if not obs_dir.exists():
        typer.echo(f"Error: {obs_dir} does not exist", err=True)
        raise typer.Exit(1)

    run = PanosetiRun(obs_dir)
    products = run.list_products()
    if not products:
        typer.echo(f"No data products found in {obs_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(products)} product(s) in {obs_dir.name}:")
    for p in products:
        seq = run.get_product(p)
        typer.echo(f"  {p}: {len(seq):,} frames")

    chunk = time_chunk if time_chunk > 0 else None
    stores = convert_run(run, out_dir, codec=codec, level=level,
                         time_chunk=chunk, shard_factor=shard_factor)

    typer.echo(f"\nWrote {len(stores)} Zarr store(s) to {out_dir}:")
    for s in stores:
        size_mb = sum(f.stat().st_size for f in s.rglob("*") if f.is_file()) / 1024**2
        typer.echo(f"  {s.name}  ({size_mb:.1f} MB)")
