"""
pypff.zarr — PFF → Zarr v3 conversion.

Requires the optional extra:
    uv sync --extra zarr
    # or: pip install pypff[zarr]

Quickstart::

    from pypff import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun("path/to/obs.pffd")
    stores = convert_run(run, "output/L0_zarr")
    # → one .zarr per (data_product, module)

Layout of each .zarr store
--------------------------
root/
  images/            (T, H, W)  — main pixel data
  unix_t_ns/         (T,)  int64  — resolved nanosecond timestamps
  headers/           — per-frame header fields, columnar
    pkt_num/  uint32
    pkt_tai/  uint16
    pkt_nsec/ uint32
    tv_sec/   int64
    tv_usec/  uint32
    quabo_num/ uint8   (ph256 single-level only)
    quabo_0/           (img/ph1024 module-level only)
      pkt_num/ uint32  ... etc.
    quabo_1/ …
    quabo_2/ …
    quabo_3/ …
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ..io2 import PanosetiRun, PFFSequence

# ── dtype table ──────────────────────────────────────────────────────────────
# Shrink header fields from the default int64 extracted by get_metadata_arrays.
# Ranges from panoseti-docs/Data-file-format.md:
#   quabo_num: 0–3 (2 bits used)
#   pkt_tai:   10-bit WR counter (0–1023)
#   pkt_num:   per-file packet counter, file ≤ 1 GB → ≤ ~10^7 frames
#   pkt_nsec:  0 – 999,999,999
#   tv_usec:   0 – 999,999
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
    """Pluggable write backend — implement this to add tensorstore or other backends."""

    def create_store(self, path: Path) -> Any:
        """Create (or overwrite) a Zarr v3 root group at *path*."""
        ...

    def create_array(
        self,
        root_group: Any,
        name: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        dimension_names: list[str] | None = None,
    ) -> Any:
        """Create an array inside *root_group*. *name* may use '/' for nesting."""
        ...

    def write_slice(
        self, array: Any, slices: tuple[slice, ...], data: np.ndarray
    ) -> None:
        """Write *data* into *array* at *slices*."""
        ...

    def set_attrs(self, obj: Any, attrs: dict[str, Any]) -> None:
        """Attach metadata attributes to a group or array."""
        ...

    def finalize(self, path: Path) -> None:
        """Flush, consolidate metadata, and close."""
        ...


# ── ZarrPythonWriter ─────────────────────────────────────────────────────────

class ZarrPythonWriter:
    """Default backend: zarr-python >= 3.0 with zstd/blosc/gzip compression."""

    def __init__(self, codec: str = "zstd", level: int = 3) -> None:
        self._codec = codec
        self._level = level

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

    def create_store(self, path: Path) -> Any:
        import shutil
        import zarr
        if path.exists():
            shutil.rmtree(path)
        return zarr.open_group(str(path), mode="w", zarr_format=3)

    def create_array(
        self,
        root_group: Any,
        name: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype,
        dimension_names: list[str] | None = None,
    ) -> Any:
        # Split nested paths: "headers/quabo_0/pkt_num" → group at "headers/quabo_0"
        parts = name.rsplit("/", 1)
        if len(parts) == 2:
            group = root_group.require_group(parts[0])
            array_name = parts[1]
        else:
            group = root_group
            array_name = name

        compressor = self._compressor()
        kwargs: dict[str, Any] = {"shape": shape, "chunks": chunks, "dtype": dtype}
        if compressor is not None:
            kwargs["compressors"] = [compressor]
        # zarr v3 reads dimension_names from array metadata; xarray uses this
        if dimension_names is not None:
            kwargs["dimension_names"] = dimension_names
        return group.create_array(array_name, **kwargs)

    def write_slice(
        self, array: Any, slices: tuple[slice, ...], data: np.ndarray
    ) -> None:
        array[slices] = data

    def set_attrs(self, obj: Any, attrs: dict[str, Any]) -> None:
        obj.attrs.update(attrs)

    def finalize(self, path: Path) -> None:
        import zarr
        try:
            zarr.consolidate_metadata(str(path))
        except Exception:
            pass  # consolidation is an optimization; don't fail the conversion


# ── PFFToZarrConverter ────────────────────────────────────────────────────────

class PFFToZarrConverter:
    """Convert a single PFFSequence to a Zarr v3 store."""

    def __init__(
        self,
        seq: "PFFSequence",
        writer: ZarrWriter | None = None,
        *,
        time_chunk: int | None = None,
        codec: str = "zstd",
        level: int = 3,
    ) -> None:
        from ..io2 import PFFSequence as _PFFSeq
        if not isinstance(seq, _PFFSeq) or seq.frame_config is None:
            raise ValueError("seq must be a fully configured PFFSequence.")
        self.seq = seq
        self.writer: ZarrWriter = writer or ZarrPythonWriter(codec=codec, level=level)
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
        return f"headers/{meta_key.replace('.', '/')}"

    def _root_attrs(self) -> dict[str, Any]:
        seq = self.seq
        conf = seq.frame_config
        assert conf is not None
        is_module = any("." in k for k in seq.metadata_offsets)
        return {
            "panoseti_pff_zarr_version": "1.0",
            "data_product": str(seq.meta.get("dp", "unknown")),
            "bytes_per_pixel": conf.bytes_per_pixel,
            "module": str(seq.meta.get("module", "unknown")),
            "source_pff_files": [p.name for p in seq.file_paths],
            "total_frames": len(seq),
            "header_format": "module" if is_module else "single",
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
        ts_chunk = C * 2

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
        header_arrays: dict[str, Any] = {}
        for key in seq.metadata_offsets:
            zarr_name = self._zarr_name(key)
            dtype = self._field_dtype(key)
            arr = writer.create_array(
                root, zarr_name, (T,), (ts_chunk,), dtype,
                dimension_names=["time"],
            )
            writer.set_attrs(arr, {"_ARRAY_DIMENSIONS": ["time"]})
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


# ── Public API ────────────────────────────────────────────────────────────────

def convert_run(
    run: "PanosetiRun",
    out_dir: str | Path,
    *,
    codec: str = "zstd",
    level: int = 3,
    time_chunk: int | None = None,
    writer: ZarrWriter | None = None,
) -> list[Path]:
    """
    Convert every data product in a PanosetiRun to a Zarr v3 store.

    Each (data_product, module) pair produces one ``.zarr`` directory under
    *out_dir*, named ``<run_dir_name>.<product_name>.zarr``.

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

    Returns
    -------
    list[Path]
        Paths of the created ``.zarr`` stores, one per product.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_base = run.run_dir.name  # e.g. "obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd"

    stores: list[Path] = []
    for product_name in run.list_products():
        seq = run.get_product(product_name)
        zarr_name = f"{run_base}.{product_name}.zarr"
        out_path = out_dir / zarr_name
        w = writer or ZarrPythonWriter(codec=codec, level=level)
        conv = PFFToZarrConverter(seq, w, time_chunk=time_chunk)
        stores.append(conv.convert(out_path))

    return stores
