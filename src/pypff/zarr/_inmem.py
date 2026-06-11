"""
pypff.zarr._inmem — In-memory L0-layout Dataset builder.

Builds an xarray Dataset whose structure mirrors exactly what
``PFFToZarrConverter.convert()`` writes to disk, so downstream code can treat
it interchangeably with ``xr.open_zarr(l0_store)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from ..io2 import PFFSequence

from ._dtypes import _HEADER_DTYPES


def _field_dtype(meta_key: str) -> np.dtype:
    leaf = meta_key.rsplit(".", 1)[-1]
    return _HEADER_DTYPES.get(leaf, np.dtype("int64"))


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
    """Build an in-memory L0-layout Dataset from a PFFSequence subrange.

    The result matches exactly what ``PFFToZarrConverter`` would write to disk
    for the same frames (same variables, dims, attrs), so downstream code can
    treat it interchangeably with ``xr.open_zarr(l0_store)``.

    Frame selection (in priority order):

    1. ``start_ns`` / ``stop_ns``: time-based, resolved via ``seq.seek_time``
       (stop_ns is exclusive — the frame at stop_ns is NOT included).
    2. ``start`` / ``stop`` / ``step``: frame-index-based (like Python slice).
    3. If all are None: the full sequence.

    Decimation via ``step`` reduces the frame count proportionally.
    """
    import xarray as xr  # lazy — zarr extra may not be installed in all envs

    conf = seq.frame_config
    if conf is None:
        raise ValueError("seq.frame_config is None — sequence has no frames or invalid format.")

    total_frames = len(seq)

    # ── 1. Resolve frame indices ──────────────────────────────────────────────

    if start_ns is not None or stop_ns is not None:
        # Time-based selection overrides index-based args.
        if start_ns is not None:
            start = seq.seek_time(start_ns)
        if stop_ns is not None:
            # stop_ns is exclusive: find the frame at/after stop_ns, but do not
            # include it.  seek_time returns the *closest* frame; we want the
            # first frame whose timestamp is >= stop_ns.
            stop_frame = seq.seek_time(stop_ns)
            # If the frame at stop_frame is strictly before stop_ns, advance by
            # one so the range is truly exclusive.
            if stop_frame < total_frames and seq.timestamp_at(stop_frame) < stop_ns:
                stop_frame += 1
            stop = stop_frame

    sl = slice(start, stop, step)
    indices = np.arange(*sl.indices(total_frames), dtype=np.int64)
    T = len(indices)

    # ── 2. Read images ────────────────────────────────────────────────────────

    if T == 0:
        images_data = np.empty((0, *conf.image_shape), dtype=conf.dtype)
    elif step is None or step == 1:
        # Contiguous range — use the fast sequential reader.
        frame_start = int(indices[0]) if T > 0 else 0
        images_data = seq.read_images_range(frame_start, T)
    else:
        # Strided or arbitrary indices.
        images_data = seq.read_images(indices)

    # ── 3. Read metadata (all header fields + unix_t_ns) ─────────────────────

    all_meta_keys = [*seq.metadata_offsets.keys(), "unix_t_ns"]

    if T == 0:
        meta_arrays: dict[str, np.ndarray] = {k: np.empty(0, dtype=np.int64) for k in all_meta_keys}
    elif step is None or step == 1:
        # Contiguous — use the efficient slice path.
        meta_arrays = seq.get_metadata_arrays(all_meta_keys, slice_obj=sl)
    else:
        # Strided — pass explicit indices.
        meta_arrays = seq.get_metadata_arrays(all_meta_keys, indices=indices)

    # ── 4. Build root attrs (mirrors _root_attrs() in PFFToZarrConverter) ────

    header_fields: list[str] = []
    quabo_fields: list[str] = []
    for key in seq.metadata_offsets:
        zarr_name = key.replace(".", "_")
        (quabo_fields if "." in key else header_fields).append(zarr_name)

    root_attrs: dict[str, Any] = {
        "panoseti_pff_zarr_version": "1.1",
        "data_product": str(seq.meta.get("dp", "unknown")),
        "bytes_per_pixel": conf.bytes_per_pixel,
        "module": str(seq.meta.get("module", "unknown")),
        "source_pff_files": [p.name for p in seq.file_paths],
        "total_frames": total_frames,  # total of seq, not the slice
        "header_format": "module" if quabo_fields else "single",
        "header_fields": sorted(header_fields),
        "quabo_fields": sorted(quabo_fields),
        "frame_config": {
            "header_size": conf.header_size,
            "payload_size": conf.payload_size,
            "frame_size": conf.frame_size,
            "image_shape": list(conf.image_shape),
            "dtype_str": conf.dtype_str,
            "bytes_per_pixel": conf.bytes_per_pixel,
            "format_name": conf.format_name,
        },
    }
    if run_configs:
        root_attrs["run_configs"] = run_configs

    # ── 5. Build xarray DataArrays ────────────────────────────────────────────

    images_da = xr.DataArray(
        images_data,
        dims=["time", "y", "x"],
        attrs={
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],
            "data_product": str(seq.meta.get("dp", "unknown")),
            "module": str(seq.meta.get("module", "unknown")),
        },
    )

    unix_t_ns_da = xr.DataArray(
        meta_arrays["unix_t_ns"].astype(np.int64),
        dims=["time"],
        attrs={
            "_ARRAY_DIMENSIONS": ["time"],
            "ns_since_epoch": "1970-01-01T00:00:00Z",
            "long_name": "UNIX timestamp in nanoseconds",
        },
    )

    data_vars: dict[str, xr.DataArray] = {
        "images": images_da,
        "unix_t_ns": unix_t_ns_da,
    }

    for key in seq.metadata_offsets:
        zarr_name = key.replace(".", "_")
        dtype = _field_dtype(key)
        raw = meta_arrays[key]
        data_vars[zarr_name] = xr.DataArray(
            raw.astype(dtype),
            dims=["time"],
            attrs={
                "_ARRAY_DIMENSIONS": ["time"],
                "pff_header_field": key,
            },
        )

    return xr.Dataset(data_vars, attrs=root_attrs)
