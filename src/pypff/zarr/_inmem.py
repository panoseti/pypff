"""_inmem.py — in-memory dataset utilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr
    from pypff.io2 import PFFSequence

def sequence_to_dataset(
    seq: PFFSequence,
    *,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    start_ns: int | None = None,
    stop_ns: int | None = None,
    run_configs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Convert a ``PFFSequence`` slice to an in-memory xarray Dataset.

    Frame selection priority:
    1. ``start_ns`` / ``stop_ns`` — time-based, stop_ns exclusive.
    2. ``start`` / ``stop`` / ``step`` — index-based (Python-slice semantics).
    3. If all are None: the full sequence.

    Parameters
    ----------
    seq:
        A fully configured ``PFFSequence``.
    start, stop, step:
        Frame-index slice (exclusive stop, same as Python slicing).
    start_ns, stop_ns:
        UNIX timestamp range in nanoseconds (exclusive stop).
    run_configs:
        Optional run-config dict embedded in the store's root attrs.

    Returns
    -------
    xr.Dataset
        In-memory dataset with all arrays (images, timestamps, header fields).
    """
    import xarray as xr
    from pypff.io2 import PFFSequence as _PFFSeq
    from pypff.zarr import PFFToZarrConverter, ZarrPythonWriter

    if not isinstance(seq, _PFFSeq) or seq.frame_config is None:
        raise ValueError("seq must be a fully configured PFFSequence.")

    # ── Build a sub-sequence view ─────────────────────────────────────────────
    # Apply time-range filter first (takes priority), then frame-index slice.
    sub: Any = seq

    if start_ns is not None or stop_ns is not None:
        # Use timestamps to find frame-index bounds.
        import numpy as np

        ts = seq.get_metadata_arrays(["unix_t_ns"])["unix_t_ns"]
        lo = int(np.searchsorted(ts, start_ns, side="left")) if start_ns is not None else 0
        hi = int(np.searchsorted(ts, stop_ns, side="left")) if stop_ns is not None else len(seq)
        sub = seq[lo:hi]
    elif start is not None or stop is not None or step is not None:
        sub = seq[slice(start, stop, step)]

    # ── Write to a temporary Zarr store, load into xarray, return ─────────────
    with tempfile.TemporaryDirectory(prefix="panoseti_s2d_") as tmp_dir:
        out_path = Path(tmp_dir) / "store.zarr"
        writer = ZarrPythonWriter(codec="none")  # no compression for in-memory speed
        conv = PFFToZarrConverter(
            sub,
            writer,
            run_configs=run_configs or {},
        )
        conv.convert(out_path)
        # load_dataset=True forces xarray to pull all data into memory before
        # the TemporaryDirectory is cleaned up.
        ds: xr.Dataset = xr.open_zarr(
            str(out_path),
            consolidated=False,
        ).load()

    return ds
