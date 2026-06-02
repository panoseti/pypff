"""
pypff.zarr — PFF → Zarr v3 conversion and read-side wrappers.

Requires the optional extra:
    uv sync --extra zarr
    # or: pip install pypff[zarr]

Write (PFF → Zarr)::

    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun("path/to/obs.pffd")
    stores = convert_run(run, "output/L0_zarr")
    # → one .zarr per (data_product, module)
    # → one .panoseti-meta/ sidecar bundle per run (configs, logs, hk.pff)

Read (Zarr → xarray / numpy)::

    from pypff.zarr import PanosetiZarrRun
    import xarray as xr

    zrun = PanosetiZarrRun("output/L0_zarr")
    store = zrun.get_product("dp_ph256.bpp_2.module_254")
    ds = store.to_dataset()          # xarray.Dataset — all arrays visible
    ts = store.timestamps()          # int64 ns array
    cfg = zrun.configs["obs_config"] # parsed obs config dict

    # Or open directly with xarray:
    ds = xr.open_zarr(str(stores[0]))  # images, unix_t_ns, pkt_num, … all visible

Layout of each .zarr store (flat root — see docs/zarr_v3_spec.md)
-----------------------------------------------------------------
All arrays live at the root of the store so that ``xr.open_zarr(store)``
surfaces every variable automatically without ``group=`` arguments or DataTree.
Logical grouping is encoded in the ``header_fields`` / ``quabo_fields`` root
attributes and the per-array ``pff_header_field`` provenance attribute.

root/
  images/            (T, H, W)  — main pixel data
  unix_t_ns/         (T,)  int64  — resolved nanosecond timestamps
  pkt_num/           (T,)  uint32  ─┐
  pkt_tai/           (T,)  uint16   │ per-frame header fields (flat at root)
  pkt_nsec/          (T,)  uint32   │
  tv_sec/            (T,)  int64    │
  tv_usec/           (T,)  uint32  ─┘
  quabo_num/         (T,)  uint8    — ph256 single-level only
  quabo_0_pkt_num/   (T,)  uint32  ─┐ img/ph1024 module-level; four quabo
  quabo_0_pkt_tai/   (T,)  uint16   │ sub-sets stored as flat names
  …                                 │ (quabo_<i>_<field>)
  quabo_3_tv_usec/   (T,)  uint32  ─┘

Run-level output layout under out_dir/
---------------------------------------
  <run>.dp_img16.bpp_2.module_1.zarr/
  <run>.dp_ph256.bpp_2.module_254.zarr/
  <run>.panoseti-meta/                   ← sidecar bundle (write_sidecars=True)
    manifest.json
    configs/   ← raw obs_config.json, daq_config.json, …
    logs/      ← log.txt, hp_stdout_*
    manifests/ ← dp_manifest*.json
    sentinels/ ← collect_complete, run_complete, recording_ended
    hk.pff
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import zarr as _zarr

    from ..io2 import PanosetiRun, PFFSequence

# Re-export read-side wrappers so callers only need to import from pypff.zarr
import contextlib

from ._reader import PanosetiZarrRun, PanosetiZarrStore, open_zarr_run

__all__ = [
    "PFFToZarrConverter",
    "PanosetiZarrRun",
    "PanosetiZarrStore",
    "ZarrPythonWriter",
    "ZarrWriter",
    "convert_run",
    "open_zarr_run",
]

# ── dtype table ──────────────────────────────────────────────────────────────
# Shrink header fields from the default int64 extracted by get_metadata_arrays.
# Ranges from panoseti-docs/Data-file-format.md:
#   quabo_num: 0-3 (2 bits used)
#   pkt_tai:   10-bit WR counter (0-1023)
#   pkt_num:   per-file packet counter, file <= 1 GB -> <= ~10^7 frames
#   pkt_nsec:  0 - 999,999,999
#   tv_usec:   0 - 999,999
#   tv_sec:    unix seconds (int64 for future-safety)
_HEADER_DTYPES: dict[str, np.dtype] = {
    "quabo_num": np.dtype("uint8"),
    "pkt_num":   np.dtype("uint32"),
    "pkt_tai":   np.dtype("uint16"),
    "pkt_nsec":  np.dtype("uint32"),
    "tv_sec":    np.dtype("int64"),
    "tv_usec":   np.dtype("uint32"),
}

_IMG_CHUNK_BYTES_TARGET = 8 * 1024 * 1024  # 8 MB pre-compression per time chunk


# ── ZarrWriter protocol ──────────────────────────────────────────────────────

@runtime_checkable
class ZarrWriter(Protocol):
    """Pluggable write backend — implement this to add tensorstore or other backends.

    Opaque handles (``root_group``, ``array``, ``obj``) are typed as ``object``
    in this Protocol because the concrete types differ per backend
    (``zarr.Group`` / ``zarr.Array`` for ``ZarrPythonWriter``, different for
    tensorstore, etc.).  Implementations cast internally; callers treat them as
    opaque tokens and only pass them back into the same writer.
    """

    def create_store(self, path: Path) -> _zarr.Group:
        """Create (or overwrite) a Zarr v3 root group at *path*."""
        ...

    def create_array(
        self,
        root_group: _zarr.Group,
        name: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        dimension_names: list[str] | None = None,
    ) -> _zarr.Array[Any]:
        """Create an array inside *root_group*. *name* may use '/' for nesting."""
        ...

    def write_slice(
        self, array: _zarr.Array[Any], slices: tuple[slice, ...], data: np.ndarray
    ) -> None:
        """Write *data* into *array* at *slices*."""
        ...

    def set_attrs(self, obj: _zarr.Group | _zarr.Array[Any], attrs: dict[str, Any]) -> None:
        """Attach metadata attributes to a group or array."""
        ...


    def finalize(self, path: Path) -> None:
        """Flush and close the store."""
        ...


# ── ZarrPythonWriter ─────────────────────────────────────────────────────────

class ZarrPythonWriter:
    """Default backend: zarr-python >= 3.0 with zstd/blosc/gzip compression."""

    def __init__(self, codec: str = "zstd", level: int = 3, *, consolidate: bool = False) -> None:
        self._codec = codec
        self._level = level
        self._consolidate = consolidate

    def _compressor(self) -> Any:
        try:
            import zarr.codecs as zc
        except ImportError as exc:
            raise ImportError(
                "zarr is not installed. Run: pip install pypff[zarr]"
            ) from exc
        if self._codec == "zstd":
            return zc.ZstdCodec(level=self._level)
        if self._codec in ("blosc-lz4", "blosc"):
            return zc.BloscCodec(cname="lz4", clevel=self._level, shuffle="shuffle")
        if self._codec == "gzip":
            return zc.GzipCodec(level=self._level)
        return None  # codec == "none"

    def create_store(self, path: Path) -> _zarr.Group:
        import shutil

        import zarr
        if path.exists():
            shutil.rmtree(path)
        return zarr.open_group(str(path), mode="w", zarr_format=3)  # type: ignore[return-value]

    def create_array(
        self,
        root_group: _zarr.Group,
        name: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        dimension_names: list[str] | None = None,
    ) -> _zarr.Array[Any]:
        import zarr
        group: zarr.Group = root_group  # type: ignore[assignment]
        # Support nested path: "a/b/c" → group "a/b", array "c"
        parts = name.rsplit("/", 1)
        if len(parts) == 2:
            group = group.require_group(parts[0])
            array_name = parts[1]
        else:
            array_name = name

        compressor = self._compressor()
        kwargs: dict[str, Any] = {"shape": shape, "chunks": chunks, "dtype": dtype}
        if compressor is not None:
            kwargs["compressors"] = [compressor]
        # zarr v3 stores dimension_names in array metadata; xarray reads this
        if dimension_names is not None:
            kwargs["dimension_names"] = dimension_names
        return group.create_array(array_name, **kwargs)  # type: ignore[return-value]

    def write_slice(
        self, array: _zarr.Array[Any], slices: tuple[slice, ...], data: np.ndarray
    ) -> None:
        array[slices] = data  # type: ignore[index]

    def set_attrs(self, obj: _zarr.Group | _zarr.Array[Any], attrs: dict[str, Any]) -> None:
        obj.attrs.update(attrs)  # type: ignore[union-attr]

    def finalize(self, path: Path) -> None:
        # Zarr v3 consolidated metadata is not part of the spec and emits
        # ZarrUserWarning in zarr-python 3.2+.  Skip by default; enable via
        # consolidate=True if open-time latency on S3 is a concern.
        if self._consolidate:
            import zarr
            with contextlib.suppress(Exception):
                zarr.consolidate_metadata(str(path))


# ── PFFToZarrConverter ────────────────────────────────────────────────────────

class PFFToZarrConverter:
    """Convert a single PFFSequence to a Zarr v3 store."""

    def __init__(
        self,
        seq: PFFSequence,
        writer: ZarrWriter | None = None,
        *,
        time_chunk: int | None = None,
        codec: str = "zstd",
        level: int = 3,
        run_configs: dict[str, Any] | None = None,
    ) -> None:
        from ..io2 import PFFSequence as _PFFSeq
        if not isinstance(seq, _PFFSeq) or seq.frame_config is None:
            raise ValueError("seq must be a fully configured PFFSequence.")
        self.seq = seq
        self.writer: ZarrWriter = writer or ZarrPythonWriter(codec=codec, level=level)
        self.run_configs = run_configs or {}
        conf = seq.frame_config
        if time_chunk is None:
            # target ~8 MB pre-compression per chunk
            bpf = max(conf.payload_size, 1)
            time_chunk = max(256, min(32768, _IMG_CHUNK_BYTES_TARGET // bpf))
        self.time_chunk = time_chunk

    def _field_dtype(self, meta_key: str) -> np.dtype:
        leaf = meta_key.rsplit(".", 1)[-1]
        return _HEADER_DTYPES.get(leaf, np.dtype("int64"))

    def _zarr_name(self, meta_key: str) -> str:
        # Store at root level so xarray.open_zarr sees all arrays as variables.
        # "pkt_num" → "pkt_num"; "quabo_0.pkt_num" → "quabo_0_pkt_num"
        return meta_key.replace(".", "_")

    def _root_attrs(self) -> dict[str, Any]:
        seq = self.seq
        conf = seq.frame_config
        assert conf is not None
        # Derive discoverability lists from metadata_offsets keys
        header_fields: list[str] = []
        quabo_fields: list[str] = []
        for key in seq.metadata_offsets:
            zarr_name = key.replace(".", "_")
            (quabo_fields if "." in key else header_fields).append(zarr_name)
        attrs: dict[str, Any] = {
            "panoseti_pff_zarr_version": "1.0",
            "data_product": str(seq.meta.get("dp", "unknown")),
            "bytes_per_pixel": conf.bytes_per_pixel,
            "module": str(seq.meta.get("module", "unknown")),
            "source_pff_files": [p.name for p in seq.file_paths],
            "total_frames": len(seq),
            "header_format": "module" if quabo_fields else "single",
            "header_fields": sorted(header_fields),
            "quabo_fields":  sorted(quabo_fields),
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
        if self.run_configs:
            attrs["run_configs"] = self.run_configs
        return attrs

    def convert(self, out_path: Path) -> Path:
        """Stream-convert the PFFSequence to a Zarr v3 store at *out_path*."""
        out_path = Path(out_path)
        seq = self.seq
        conf = seq.frame_config
        assert conf is not None
        writer = self.writer

        T = len(seq)
        H, W = conf.image_shape
        C = self.time_chunk
        # Cap 1D chunks at 65536 frames (512 KB for int64, scales down for narrower dtypes).
        # The divisor 8 targets the widest dtype (int64) so the cap is conservative for all
        # current fields. Floor at C keeps ts_chunk stride-aligned with image chunks.
        ts_chunk = max(C, min(65536, _IMG_CHUNK_BYTES_TARGET // 8))

        root = writer.create_store(out_path)
        writer.set_attrs(root, self._root_attrs())

        # ── images ──────────────────────────────────────────────
        img_arr = writer.create_array(
            root, "images", (T, H, W), (C, H, W), conf.dtype,
            dimension_names=["time", "y", "x"],
        )
        writer.set_attrs(img_arr, {
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],  # zarr v2 compat
            "data_product": str(seq.meta.get("dp", "unknown")),
            "module": str(seq.meta.get("module", "unknown")),
        })

        # ── unix_t_ns ────────────────────────────────────────────
        ts_arr = writer.create_array(
            root, "unix_t_ns", (T,), (ts_chunk,), np.dtype("int64"),
            dimension_names=["time"],
        )
        writer.set_attrs(ts_arr, {
            "_ARRAY_DIMENSIONS": ["time"],
            # Use a non-CF units name so xarray doesn't try CF time-decoding.
            # Access as raw int64 or convert: arr.values.view("datetime64[ns]")
            "ns_since_epoch": "1970-01-01T00:00:00Z",
            "long_name": "UNIX timestamp in nanoseconds",
        })

        # ── header columns ───────────────────────────────────────
        # Arrays are at the root level of the store (flat names) so that
        # xarray.open_zarr sees them as dataset variables automatically.
        header_arrays: dict[str, _zarr.Array[Any]] = {}
        for key in seq.metadata_offsets:
            zarr_name = self._zarr_name(key)  # e.g. "pkt_num", "quabo_0_pkt_num"
            dtype = self._field_dtype(key)
            arr = writer.create_array(
                root, zarr_name, (T,), (ts_chunk,), dtype,
                dimension_names=["time"],
            )
            writer.set_attrs(arr, {
                "_ARRAY_DIMENSIONS": ["time"],
                "pff_header_field": key,  # original dotted key for provenance
            })
            header_arrays[key] = arr

        # ── stream-convert ───────────────────────────────────────
        t_start = 0
        for batch, meta, ts in seq.iter_batches(
            size=C, with_headers=True, with_timestamps=True
        ):
            n = batch.shape[0]
            sl1d = (slice(t_start, t_start + n),)
            sl3d = (slice(t_start, t_start + n), slice(None), slice(None))

            writer.write_slice(img_arr, sl3d, np.asarray(batch))
            writer.write_slice(ts_arr, sl1d, ts)

            for key, arr in header_arrays.items():
                if key in meta:
                    writer.write_slice(arr, sl1d, meta[key].astype(self._field_dtype(key)))

            t_start += n

        writer.finalize(out_path)
        return out_path


# ── Sidecar bundle helpers ────────────────────────────────────────────────────

def _extract_run_configs(run: PanosetiRun) -> dict[str, Any]:
    """Extract parsed run configs from a PanosetiRun as plain JSON-serializable dicts."""
    result: dict[str, Any] = {}
    try:
        for name, model in run.configs.items():
            try:
                if hasattr(model, "model_dump"):
                    # mode="json" converts non-serializable types (IPv4Address, etc.)
                    # to their JSON-safe equivalents (strings, numbers, lists).
                    result[name] = model.model_dump(mode="json")
                elif isinstance(model, dict):
                    result[name] = model
            except Exception:
                pass
    except Exception:
        pass
    return result


def _write_sidecar_bundle(
    run: PanosetiRun,
    meta_dir: Path,
    zarr_stores: list[Path],
) -> None:
    """Copy ancillary run files into *meta_dir* and write a manifest.json index."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run.run_dir

    def _copy_glob(pattern: str, dest: Path) -> None:
        dest.mkdir(exist_ok=True)
        for f in run_dir.glob(pattern):
            if f.is_file():
                shutil.copy2(f, dest / f.name)

    _copy_glob("*.json", meta_dir / "configs")
    _copy_glob("*.toml", meta_dir / "configs")
    _copy_glob("*.log",  meta_dir / "logs")
    _copy_glob("*.txt",  meta_dir / "logs")
    _copy_glob("hp_stdout_*", meta_dir / "logs")
    _copy_glob("dp_manifest*", meta_dir / "manifests")

    sentinels_dir = meta_dir / "sentinels"
    sentinels_dir.mkdir(exist_ok=True)
    for name in ("collect_complete", "run_complete", "recording_ended"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, sentinels_dir / name)

    hk = run_dir / "hk.pff"
    if hk.exists():
        shutil.copy2(hk, meta_dir / "hk.pff")

    manifest: dict[str, Any] = {
        "panoseti_meta_version": "1.0",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_run_dir": str(run_dir),
        "zarr_stores": [s.name for s in zarr_stores],
    }
    (meta_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── Public API ────────────────────────────────────────────────────────────────

def convert_run(
    run: PanosetiRun,
    out_dir: str | Path,
    *,
    codec: str = "zstd",
    level: int = 3,
    time_chunk: int | None = None,
    writer: ZarrWriter | None = None,
    write_sidecars: bool = True,
    embed_configs: bool = True,
) -> list[Path]:
    """
    Convert every data product in a PanosetiRun to a Zarr v3 store.

    Each (data_product, module) pair produces one ``.zarr`` directory under
    *out_dir*, named ``<run_dir_name>.<product_name>.zarr``.  By default a
    sibling ``<run_dir_name>.panoseti-meta/`` directory is also written with
    copies of all non-PFF ancillary files (configs, logs, manifests, hk.pff).

    Parameters
    ----------
    run:
        A ``PanosetiRun`` pointing to a ``.pffd`` observation directory.
    out_dir:
        Directory where ``.zarr`` stores will be written.
    codec:
        Compression codec — ``"zstd"`` (default), ``"blosc-lz4"``, ``"gzip"``,
        or ``"none"``.
    level:
        Compression level (codec-specific; 3 is a good default for zstd).
    time_chunk:
        Frames per time chunk. Auto-sized to ~8 MB pre-compression if omitted.
    writer:
        Custom ``ZarrWriter`` backend. Defaults to ``ZarrPythonWriter``.
    write_sidecars:
        Copy ancillary run files (configs, logs, manifests, hk.pff) to a
        sibling ``<run>.panoseti-meta/`` directory.  Set to ``False`` for
        Nextflow channels where only the ``.zarr`` directories should appear.
    embed_configs:
        Embed parsed run configs as ``run_configs`` in each store's root attrs,
        making each ``.zarr`` self-contained for analysis.

    Returns
    -------
    list[Path]
        Paths of the created ``.zarr`` stores, one per product.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_base = run.run_dir.name

    run_configs = _extract_run_configs(run) if embed_configs else {}

    stores: list[Path] = []
    for product_name in run.list_products():
        seq = run.get_product(product_name)
        zarr_name = f"{run_base}.{product_name}.zarr"
        out_path = out_dir / zarr_name
        w = writer or ZarrPythonWriter(codec=codec, level=level)
        conv = PFFToZarrConverter(seq, w, time_chunk=time_chunk, run_configs=run_configs)
        stores.append(conv.convert(out_path))

    if write_sidecars:
        meta_dir = out_dir / f"{run_base}.panoseti-meta"
        _write_sidecar_bundle(run, meta_dir, stores)

    return stores
