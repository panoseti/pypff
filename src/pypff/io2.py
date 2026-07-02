from __future__ import annotations

import bisect
import contextlib
import logging
import mmap
import re
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
import orjson
from pydantic import ValidationError

from .models import FrameConfig, ModuleHeader, QuaboHeader
from .utils import extract_seqno, parse_filename

logger = logging.getLogger(__name__)

# Sentinel for the derived unix_t_ns virtual metadata key
_UNIX_T_NS = "unix_t_ns"

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _parse_hk_file(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse an hk.pff file into {device: {field: np.ndarray}}."""
    key_map = {"TEMP1": "DET_TEMP", "TEMP2": "FPGA_TEMP"}
    raw: dict[str, dict[str, list[Any]]] = {}

    if not path.exists() or path.stat().st_size == 0:
        return {}

    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            return {}

        start, size = 0, mm.size()
        while start < size:
            end = mm.find(b"\n\n", start)
            if end == -1:
                chunk, start = mm[start:], size
            else:
                chunk, start = mm[start:end], end + 2
            if not chunk.strip():
                continue
            try:
                for top_key, values in orjson.loads(chunk).items():
                    target = raw.setdefault(top_key, {})
                    for k, v in values.items():
                        canon = key_map.get(k, k)
                        if canon not in target:
                            target[canon] = []
                        if isinstance(v, str):
                            with contextlib.suppress(ValueError):
                                v = float(v) if "." in v else int(v)
                        target[canon].append(v)
            except (orjson.JSONDecodeError, ValueError, AttributeError):
                continue
        mm.close()

    return {
        dev: {k: np.asarray(vs) for k, vs in fields.items()}
        for dev, fields in raw.items()
    }


# ─────────────────────────────────────────────────────────────
#  LRU mmap cache
# ─────────────────────────────────────────────────────────────

class _MmapLRU:
    """Fixed-capacity LRU cache for open mmap handles."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._mmaps: OrderedDict[int, mmap.mmap] = OrderedDict()
        self._files: dict[int, Any] = {}

    def get(self, idx: int, path: Path) -> mmap.mmap | None:
        if idx in self._mmaps:
            self._mmaps.move_to_end(idx)
            return self._mmaps[idx]
        try:
            f = open(path, "rb")  # noqa: SIM115
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):
            return None
        self._files[idx] = f
        self._mmaps[idx] = mm
        if len(self._mmaps) > self.capacity:
            evict_idx, evict_mm = self._mmaps.popitem(last=False)
            evict_mm.close()
            self._files.pop(evict_idx).close()
        return mm

    def close_all(self) -> None:
        for mm in self._mmaps.values():
            mm.close()
        for f in self._files.values():
            f.close()
        self._mmaps.clear()
        self._files.clear()

    def __len__(self) -> int:
        return len(self._mmaps)

    def __getstate__(self) -> dict[str, Any]:
        return {"capacity": self.capacity}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.capacity = state["capacity"]
        self._mmaps = OrderedDict()
        self._files = {}


# ─────────────────────────────────────────────────────────────
#  PFFSequence
# ─────────────────────────────────────────────────────────────

class PFFSequence:
    """
    Time-ordered sequence of PFF files for a single data product.

    Zero-copy access via mmap. Streaming is the default; slicing and random
    access are also supported but may allocate.

    Use as a context manager to ensure file handles are released::

        with PFFSequence(paths) as seq:
            for batch in seq.iter_batches(size=256):
                process(batch)
    """

    def __init__(self, file_paths: Sequence[str | Path], max_open_files: int = 16) -> None:
        paths = [Path(p) for p in file_paths]
        if not paths:
            raise ValueError("No files provided.")

        self.file_paths = sorted(paths, key=extract_seqno)
        self.name = self.file_paths[0].name.split(".seqno")[0]
        self.meta = parse_filename(self.file_paths[0].name)

        self.header_size: int = 0
        self.frame_config: FrameConfig | None = None
        self._file_frame_counts: list[int] = []
        self._cumulative_frames: list[int] = []
        self._total_frames: int = 0

        # Per-file coarse (first_ts, last_ts) for binary search; None until indexed
        self._file_ts_bounds: list[tuple[int, int] | None] = []
        # Per-file dense timestamp arrays; populated lazily by seek_time
        self._file_timestamps: list[np.ndarray | None] = []
        # Full-sequence timestamp cache (populated by timestamps())
        self._all_timestamps: np.ndarray | None = None

        self.metadata_offsets: dict[str, tuple[int, int]] = {}
        self._lru = _MmapLRU(max_open_files)

        self._analyze_structure()
        self._index_files()

    # ── context manager ──────────────────────────────────────

    def __enter__(self) -> PFFSequence:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly release all open file handles and mmap regions."""
        self._lru.close_all()

    # ── pickle support ───────────────────────────────────────

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_lru"] = _MmapLRU(self._lru.capacity)
        state["_all_timestamps"] = None
        state["_file_timestamps"] = [None] * len(self._file_timestamps)
        state["_file_ts_bounds"] = list(self._file_ts_bounds)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # ── length / iteration ───────────────────────────────────

    def __len__(self) -> int:
        return self._total_frames

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield zero-copy strided views of each frame in sequence order."""
        return self.iter_batches(size=1, _yield_single=True)  # type: ignore[return-value]

    # ── indexing ─────────────────────────────────────────────

    @overload
    def __getitem__(self, key: int) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: slice) -> np.ndarray: ...

    def __getitem__(self, key: int | slice) -> np.ndarray:
        if isinstance(key, (int, np.integer)):
            key = int(key)
            if key < 0:
                key += self._total_frames
            if not (0 <= key < self._total_frames):
                raise IndexError("Index out of range")
            return self._frame_view(key)
        elif isinstance(key, slice):
            indices = np.arange(*key.indices(self._total_frames), dtype=np.int64)
            if len(indices) == 0:
                conf = self.frame_config
                shape = conf.image_shape if conf else (0, 0)
                dtype = conf.dtype if conf else np.uint8
                return np.empty((0, *shape), dtype=dtype)
            return self.read_images(indices)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    # ── structure analysis ───────────────────────────────────

    def _analyze_structure(self) -> None:
        sample = next((p for p in self.file_paths if p.stat().st_size > 0), None)
        if not sample:
            return

        with open(sample, "rb") as f:
            chunk = f.read(4096)
            match = re.search(b"}\n\n\\*", chunk) or re.search(b"\n\n\\*", chunk)
            if not match:
                raise ValueError(f"Invalid PFF format in {sample}")

            self.header_size = match.end() - 1
            self._analyze_offsets(chunk[: self.header_size])

            fmt = str(self.meta.get("dp", "unknown")).lower()
            shape: tuple[int, int]
            dtype: Any
            bpp: int
            if "img8" in fmt:
                shape, dtype, bpp = (32, 32), np.uint8, 1
            elif "img16" in fmt:
                shape, dtype, bpp = (32, 32), np.uint16, 2
            elif "ph256" in fmt:
                shape, dtype, bpp = (16, 16), np.int16, 2
            else:
                shape, dtype, bpp = (32, 32), np.int16, 2

            payload_size = shape[0] * shape[1] * bpp
            self.frame_config = FrameConfig(
                header_size=self.header_size,
                payload_size=payload_size,
                frame_size=self.header_size + 1 + payload_size,
                image_shape=shape,
                dtype_str=np.dtype(dtype).name,
                bytes_per_pixel=bpp,
                format_name=fmt,
            )

    def _analyze_offsets(self, header_bytes: bytes) -> None:
        self.metadata_offsets = {}
        q_pos = [header_bytes.find(f'"quabo_{i}"'.encode()) for i in range(4)]
        is_module = q_pos[0] != -1

        pattern = re.compile(rb'"(\w+)":([^,{}]+)')
        for m in pattern.finditer(header_bytes):
            key = m.group(1).decode()
            val_bytes = m.group(2)
            if not re.match(rb"^\s*-?\d+\s*$", val_bytes):
                continue
            start, end = m.start(2), m.end(2)
            if is_module:
                q_idx = next(
                    (i for i in range(3, -1, -1) if q_pos[i] != -1 and q_pos[i] < m.start()),
                    -1,
                )
                if q_idx != -1:
                    self.metadata_offsets[f"quabo_{q_idx}.{key}"] = (start, end)
            else:
                self.metadata_offsets[key] = (start, end)

    def _index_files(self) -> None:
        if not self.frame_config:
            return
        conf = self.frame_config
        total = 0
        for p in self.file_paths:
            size = p.stat().st_size
            # Validate header_size consistency for every file
            if size > 0:
                with open(p, "rb") as f:
                    probe = f.read(min(4096, size))
                match = re.search(b"}\n\n\\*", probe) or re.search(b"\n\n\\*", probe)
                if not match or (match.end() - 1) != conf.header_size:
                    raise ValueError(
                        f"Frame structure mismatch in {p.name}: "
                        f"expected header_size={conf.header_size}, found "
                        f"{(match.end() - 1) if match else 'no marker'}"
                    )
            n = size // conf.frame_size if size > 0 else 0
            self._file_frame_counts.append(n)
            self._cumulative_frames.append(total + n)
            self._file_ts_bounds.append(None)
            self._file_timestamps.append(None)
            total += n
        self._total_frames = total

    # ── internal helpers ──────────────────────────────────────

    def _get_mmap(self, file_idx: int) -> mmap.mmap | None:
        return self._lru.get(file_idx, self.file_paths[file_idx])

    def _locate_frame(self, idx: int) -> tuple[int, int]:
        if not (0 <= idx < self._total_frames):
            raise IndexError(f"Frame {idx} out of range [0, {self._total_frames})")
        file_idx = bisect.bisect_right(self._cumulative_frames, idx)
        prev = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
        return file_idx, idx - prev

    def _frame_view(self, global_idx: int) -> np.ndarray:
        """Return a zero-copy strided view of a single frame's image data."""
        if not self.frame_config:
            raise RuntimeError("Frame config missing.")
        conf = self.frame_config
        file_idx, local_idx = self._locate_frame(global_idx)
        mm = self._get_mmap(file_idx)
        if mm is None:
            raise RuntimeError(f"Cannot open mmap for file {file_idx}")
        offset = local_idx * conf.frame_size + conf.header_size + 1
        return np.ndarray(
            shape=conf.image_shape,
            dtype=conf.dtype,
            buffer=mm,
            offset=offset,
            strides=np.empty(conf.image_shape, dtype=conf.dtype).strides,
        )

    # ── streaming iteration ───────────────────────────────────

    def iter_batches(
        self,
        size: int = 256,
        *,
        with_headers: bool = False,
        with_timestamps: bool = False,
        _yield_single: bool = False,
    ) -> Iterator[np.ndarray | tuple[Any, ...]]:
        """
        Yield batches of frames in sequential order.

        Within a single file, yields zero-copy strided views when possible.
        Copies only when a batch crosses a file boundary.

        Parameters
        ----------
        size:
            Maximum frames per batch.
        with_headers:
            If True, yield ``(images, header_dict_array)`` tuples where
            ``header_dict_array`` is a dict mapping field name → np.ndarray.
        with_timestamps:
            If True, yield ``(images, unix_t_ns_array)`` tuples.
            Compatible with ``with_headers`` (yields a 3-tuple).
        """
        if not self.frame_config or self._total_frames == 0:
            return

        conf = self.frame_config
        base_strides = np.empty(conf.image_shape, dtype=conf.dtype).strides
        mmap_strides = (conf.frame_size, *base_strides)
        img_offset = conf.header_size + 1

        need_meta = with_headers or with_timestamps
        ts_keys = self._timing_keys()

        global_start = 0
        for file_idx, n_frames in enumerate(self._file_frame_counts):
            if n_frames == 0:
                continue
            mm = self._get_mmap(file_idx)
            if mm is None:
                global_start += n_frames
                continue

            local_start = 0
            while local_start < n_frames:
                chunk = min(size, n_frames - local_start)
                offset = local_start * conf.frame_size + img_offset

                view = np.ndarray(
                    shape=(chunk, *conf.image_shape),
                    dtype=conf.dtype,
                    buffer=mm,
                    offset=offset,
                    strides=mmap_strides,
                )

                if not need_meta:
                    if _yield_single:
                        for i in range(chunk):
                            yield view[i]
                    else:
                        yield view
                else:
                    g_indices = np.arange(global_start + local_start,
                                          global_start + local_start + chunk, dtype=np.int64)
                    if with_headers and with_timestamps:
                        meta = self._extract_meta_sequential_file(file_idx, local_start, chunk)
                        ts = self._derive_unix_t_ns(meta, ts_keys)
                        yield view, meta, ts
                    elif with_timestamps:
                        meta = self._extract_meta_sequential_file(file_idx, local_start, chunk)
                        ts = self._derive_unix_t_ns(meta, ts_keys)
                        yield view, ts
                    else:
                        meta = self._extract_meta_sequential_file(file_idx, local_start, chunk)
                        yield view, meta
                    _ = g_indices  # consumed above

                local_start += chunk
            global_start += n_frames

    def iter_byte_range(
        self,
        file_idx: int,
        byte_start: int,
        byte_end: int,
        batch_size: int = 256,
        *,
        with_headers: bool = False,
        with_timestamps: bool = False,
    ) -> Iterator[np.ndarray | tuple[Any, ...]]:
        """
        Yield batches from a byte-aligned subrange of a single file.

        Frame-aligns ``byte_start`` and ``byte_end`` automatically. Designed
        for distributed workers (e.g., Dask) that split files by byte offset.

        Parameters
        ----------
        file_idx:
            Index into ``self.file_paths``.
        byte_start, byte_end:
            Raw byte offsets into the file; will be snapped to frame boundaries.
        batch_size:
            Maximum frames per batch.
        with_headers:
            Yield ``(images, header_fields)`` tuples.
        with_timestamps:
            Yield ``(images, unix_t_ns)`` tuples.
        """
        if not self.frame_config:
            return
        conf = self.frame_config
        fs = conf.frame_size

        local_start = byte_start // fs
        local_end = min((byte_end + fs - 1) // fs, self._file_frame_counts[file_idx])
        n = local_end - local_start
        if n <= 0:
            return

        base_strides = np.empty(conf.image_shape, dtype=conf.dtype).strides
        mmap_strides = (fs, *base_strides)
        img_offset = conf.header_size + 1
        ts_keys = self._timing_keys()

        mm = self._get_mmap(file_idx)
        if mm is None:
            return

        pos = local_start
        while pos < local_end:
            chunk = min(batch_size, local_end - pos)
            offset = pos * fs + img_offset
            view = np.ndarray(
                shape=(chunk, *conf.image_shape),
                dtype=conf.dtype,
                buffer=mm,
                offset=offset,
                strides=mmap_strides,
            )
            if not (with_headers or with_timestamps):
                yield view
            else:
                meta = self._extract_meta_sequential_file(file_idx, pos, chunk)
                ts = self._derive_unix_t_ns(meta, ts_keys) if with_timestamps else None
                if with_headers and with_timestamps:
                    yield view, meta, ts
                elif with_timestamps:
                    yield view, ts
                else:
                    yield view, meta
            pos += chunk

    # ── metadata extraction ───────────────────────────────────

    def _timing_keys(self) -> list[str]:
        """Return the correct prefix for timing fields based on header format."""
        if "quabo_0.tv_sec" in self.metadata_offsets:
            return ["quabo_0.tv_sec", "quabo_0.tv_usec", "quabo_0.pkt_nsec", "quabo_0.pkt_tai"]
        return ["tv_sec", "tv_usec", "pkt_nsec", "pkt_tai"]

    def _extract_meta_sequential_file(
        self, file_idx: int, local_start: int, count: int
    ) -> dict[str, np.ndarray]:
        """Single-pass composite-dtype extraction for a contiguous block within one file."""
        if not self.frame_config or not self.metadata_offsets:
            return {}
        conf = self.frame_config
        mm = self._get_mmap(file_idx)
        if mm is None:
            return {k: np.zeros(count, dtype=np.int64) for k in self.metadata_offsets}

        keys = list(self.metadata_offsets.keys())
        return self._composite_extract(mm, keys, local_start, count, conf)

    def _composite_extract(
        self,
        mm: mmap.mmap,
        keys: list[str],
        local_start: int,
        count: int,
        conf: FrameConfig,
    ) -> dict[str, np.ndarray]:
        """
        Build one composite structured dtype covering all requested (start, end)
        slots and issue a single np.frombuffer call per file chunk.

        Keys are sorted by byte offset before dtype construction so gaps are
        always non-negative.
        """
        valid_keys = [k for k in keys if k in self.metadata_offsets]
        results: dict[str, np.ndarray] = {}

        if not valid_keys:
            return {k: np.zeros(count, dtype=np.int64) for k in keys}

        # Sort keys by their start offset so gaps are non-negative
        valid_keys_sorted = sorted(valid_keys, key=lambda k: self.metadata_offsets[k][0])

        fs = conf.frame_size
        names = [f"f{i}" for i in range(len(valid_keys_sorted))]
        formats = [
            f"S{self.metadata_offsets[k][1] - self.metadata_offsets[k][0]}"
            for k in valid_keys_sorted
        ]
        offsets = [self.metadata_offsets[k][0] for k in valid_keys_sorted]

        dt = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": fs})
        raw = np.frombuffer(mm, dtype=dt, count=count, offset=local_start * fs)

        for i, key in enumerate(valid_keys_sorted):
            field = raw[f"f{i}"]
            # Decode fixed-width ASCII byte strings to unicode then parse as int64.
            # np.char.decode converts S{w} → U{w}; np.char.strip removes whitespace padding;
            # .astype(np.int64) handles both positive and negative integer strings.
            results[key] = np.char.strip(np.char.decode(field, "ascii")).astype(np.int64)

        return results

    def _derive_unix_t_ns(
        self, meta: dict[str, np.ndarray], ts_keys: list[str]
    ) -> np.ndarray:
        """Vectorized precise timestamp from raw header fields."""
        prefix = "quabo_0." if ts_keys[0].startswith("quabo_0") else ""
        tv_sec = meta.get(f"{prefix}tv_sec", np.zeros(1, dtype=np.int64))
        tv_usec = meta.get(f"{prefix}tv_usec", np.zeros(1, dtype=np.int64))
        pkt_nsec = meta.get(f"{prefix}pkt_nsec", np.zeros(1, dtype=np.int64))
        pkt_tai = meta.get(f"{prefix}pkt_tai", np.zeros(1, dtype=np.int64))

        n = len(tv_sec)
        final_sec = tv_sec.copy()

        has_tai = pkt_tai != 0
        if np.any(has_tai):
            d = (tv_sec - pkt_tai + 37) % 1024
            final_sec[has_tai & (d == 1)] -= 1
            final_sec[has_tai & (d == 1023)] += 1

        no_tai = ~has_tai
        if np.any(no_tai):
            tv_ns = tv_usec[no_tai] * 1_000
            diff = tv_ns - pkt_nsec[no_tai]
            sub = final_sec[no_tai]
            sub[diff > 500_000_000] += 1
            sub[diff < -500_000_000] -= 1
            final_sec[no_tai] = sub

        return (final_sec * 1_000_000_000) + pkt_nsec

    def get_metadata_arrays(
        self,
        keys: list[str],
        indices: Sequence[int] | np.ndarray | None = None,
        slice_obj: slice | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Extract metadata fields for the requested frames.

        Uses a single composite-dtype np.frombuffer pass per file (not one per key).
        Supports the derived virtual key ``'unix_t_ns'`` for precise nanosecond timestamps.

        Parameters
        ----------
        keys:
            Field names from ``metadata_offsets``, or the virtual ``'unix_t_ns'``.
        indices:
            Specific frame indices. Access is sorted for disk locality and
            results are returned in the caller's original order.
        slice_obj:
            Slice object; converted to an index array internally.
        """
        if not self.frame_config:
            return {}
        conf = self.frame_config

        # Separate virtual key from real keys
        want_ts = _UNIX_T_NS in keys
        real_keys = [k for k in keys if k != _UNIX_T_NS]
        ts_keys = self._timing_keys()
        if want_ts:
            for tk in ts_keys:
                if tk not in real_keys:
                    real_keys.append(tk)

        if slice_obj is not None:
            indices = np.arange(*slice_obj.indices(self._total_frames), dtype=np.int64)

        # ── INDEXED ACCESS PATH ───────────────────────────────
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            count = len(indices)
            out: dict[str, np.ndarray] = {k: np.zeros(count, dtype=np.int64) for k in keys}
            if count == 0:
                return out

            sort_order = np.argsort(indices, kind="stable")
            sorted_idx = indices[sort_order]
            inv = np.empty_like(sort_order)
            inv[sort_order] = np.arange(count, dtype=np.int64)

            # Group by file for bulk reads — use _composite_extract (vectorized frombuffer)
            # over the contiguous span [min_local, max_local] and then fancy-index into it.
            raw_results: dict[str, np.ndarray] = {k: np.zeros(count, dtype=np.int64) for k in real_keys}
            for file_idx, _n_frames in enumerate(self._file_frame_counts):
                file_start = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
                file_end = self._cumulative_frames[file_idx]
                mask = (sorted_idx >= file_start) & (sorted_idx < file_end)
                if not np.any(mask):
                    continue
                local_idx = sorted_idx[mask] - file_start
                mm = self._get_mmap(file_idx)
                if mm is None:
                    continue
                # Read the contiguous span from min→max local index in one frombuffer call,
                # then select only the rows we actually need via fancy indexing.
                lo, hi = int(local_idx.min()), int(local_idx.max()) + 1
                batch = self._composite_extract(mm, real_keys, lo, hi - lo, conf)
                where_mask = np.where(mask)[0]
                for key in real_keys:
                    if key in batch:
                        raw_results[key][where_mask] = batch[key][local_idx - lo]

            # Re-order to caller's original order
            for k in real_keys:
                src_key = k if k in keys else None
                if src_key:
                    out[k] = raw_results[k][inv]

            if want_ts:
                ts_meta = {k: raw_results[k][inv] for k in ts_keys if k in raw_results}
                out[_UNIX_T_NS] = self._derive_unix_t_ns(ts_meta, ts_keys)

            return {k: out[k] for k in keys}

        # ── SEQUENTIAL BULK PATH ─────────────────────────────
        out = {k: np.zeros(self._total_frames, dtype=np.int64) for k in keys}
        raw: dict[str, np.ndarray] = {}
        frames_done = 0
        for file_idx, n_frames in enumerate(self._file_frame_counts):
            if n_frames == 0:
                continue
            mm = self._get_mmap(file_idx)
            if mm is None:
                frames_done += n_frames
                continue
            batch = self._composite_extract(mm, real_keys, 0, n_frames, conf)
            for k, v in batch.items():
                if k not in raw:
                    raw[k] = np.empty(self._total_frames, dtype=np.int64)
                raw[k][frames_done : frames_done + n_frames] = v
            frames_done += n_frames

        for k in real_keys:
            if k in keys and k in raw:
                out[k] = raw[k]

        if want_ts:
            ts_meta = {k: raw.get(k, np.zeros(self._total_frames, dtype=np.int64)) for k in ts_keys}
            out[_UNIX_T_NS] = self._derive_unix_t_ns(ts_meta, ts_keys)

        return {k: out[k] for k in keys}

    def get_all_metadata(self) -> dict[str, Any]:
        """Return all discovered metadata fields for the entire sequence."""
        flat = self.get_metadata_arrays(list(self.metadata_offsets.keys()))
        structured: dict[str, Any] = {}
        for k, v in flat.items():
            if "." in k:
                parent, child = k.split(".", 1)
                if parent not in structured:
                    structured[parent] = {}
                structured[parent][child] = v
            else:
                structured[k] = v
        return structured

    # ── image access ─────────────────────────────────────────

    def read_images(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """
        Return image frames for the given indices as a contiguous (N, H, W) array.

        Indices are sorted internally for disk locality; the result is returned
        in the caller's original order.
        """
        if not self.frame_config:
            return np.empty(0)
        conf = self.frame_config
        indices = np.asarray(indices, dtype=np.int64)
        n = len(indices)
        if n == 0:
            return np.empty((0, *conf.image_shape), dtype=conf.dtype)

        result = np.empty((n, *conf.image_shape), dtype=conf.dtype)
        sort_order = np.argsort(indices, kind="stable")
        sorted_idx = indices[sort_order]

        img_offset = conf.header_size + 1
        base_strides = np.empty(conf.image_shape, dtype=conf.dtype).strides

        for res_i, global_idx in enumerate(sorted_idx):
            if not (0 <= global_idx < self._total_frames):
                continue
            file_idx, local_idx = self._locate_frame(global_idx)
            mm = self._get_mmap(file_idx)
            if mm is None:
                continue
            offset = local_idx * conf.frame_size + img_offset
            # sort_order[res_i] is the position in the caller's original indices array
            result[sort_order[res_i]] = np.ndarray(
                shape=conf.image_shape,
                dtype=conf.dtype,
                buffer=mm,
                offset=offset,
                strides=base_strides,
            )
        return result

    def read_images_range(self, start: int = 0, count: int | None = None) -> np.ndarray:
        """
        Return a contiguous (N, H, W) array for sequential frames [start, start+count).

        Unlike slicing, this always allocates and copies — use ``iter_batches``
        when you don't need the whole range in RAM at once.
        """
        if not self.frame_config:
            return np.empty(0)
        conf = self.frame_config
        count = min(count if count is not None else self._total_frames - start, self._total_frames - start)
        if count <= 0:
            return np.empty((0, *conf.image_shape), dtype=conf.dtype)

        result = np.empty((count, *conf.image_shape), dtype=conf.dtype)
        base_strides = np.empty(conf.image_shape, dtype=conf.dtype).strides
        mmap_strides = (conf.frame_size, *base_strides)
        img_offset = conf.header_size + 1

        collected, cur = 0, start
        while collected < count:
            file_idx, local_start = self._locate_frame(cur)
            to_read = min(count - collected, self._file_frame_counts[file_idx] - local_start)
            if to_read > 0:
                mm = self._get_mmap(file_idx)
                if mm:
                    view = np.ndarray(
                        shape=(to_read, *conf.image_shape),
                        dtype=conf.dtype,
                        buffer=mm,
                        offset=local_start * conf.frame_size + img_offset,
                        strides=mmap_strides,
                    )
                    result[collected : collected + to_read] = view
            collected += to_read
            cur += to_read

        return result

    # ── frame header access ───────────────────────────────────

    def get_frame(self, idx: int) -> tuple[dict[str, Any], np.ndarray]:
        """
        Return ``(header_dict, image_view)`` for the given frame.

        Header is a raw dict for speed. Use ``get_frame_validated`` if you need
        Pydantic attribute validation.
        """
        if not self.frame_config:
            raise RuntimeError("Frame config missing.")
        conf = self.frame_config
        file_idx, local_idx = self._locate_frame(idx)
        mm = self._get_mmap(file_idx)
        if mm is None:
            raise RuntimeError(f"Cannot open mmap for file {file_idx}")
        offset = local_idx * conf.frame_size
        header_dict: dict[str, Any] = orjson.loads(mm[offset : offset + conf.header_size])
        img_view = self._frame_view(idx)
        return header_dict, img_view

    def get_frame_validated(self, idx: int) -> tuple[QuaboHeader | ModuleHeader | dict[str, Any], np.ndarray]:
        """
        Return ``(validated_header, image_view)`` where the header is a Pydantic object.

        Slower than ``get_frame`` due to Pydantic validation. Use for one-off inspection,
        not in tight loops.
        """
        h_dict, img = self.get_frame(idx)
        if "quabo_0" in h_dict:
            header: QuaboHeader | ModuleHeader | dict[str, Any] = ModuleHeader(**h_dict)
        elif "quabo_num" in h_dict:
            header = QuaboHeader(**h_dict)
        else:
            header = h_dict
        return header, img

    # ── timestamp API ─────────────────────────────────────────

    def timestamps(
        self,
        indices: Sequence[int] | np.ndarray | None = None,
        *,
        as_datetime: bool = False,
    ) -> np.ndarray:
        """
        Return precise nanosecond timestamps as ``int64`` (default) or ``datetime64[ns]``.

        ``datetime64[ns]`` is a zero-copy view of the same ``int64`` data and integrates
        natively with matplotlib date axes, pandas, and xarray. Use it for display; keep
        ``int64`` for arithmetic to stay in integer space.

        With no arguments, returns the full sequence (cached after first call).
        With ``indices``, extracts only those frames (not cached).
        """
        if indices is not None:
            ts = self.get_metadata_arrays([_UNIX_T_NS], indices=np.asarray(indices, dtype=np.int64))[_UNIX_T_NS]
        elif self._all_timestamps is not None:
            ts = self._all_timestamps
        else:
            self._all_timestamps = self.get_metadata_arrays([_UNIX_T_NS])[_UNIX_T_NS]
            ts = self._all_timestamps
        return ts.view("datetime64[ns]") if as_datetime else ts

    def timestamp_at(self, idx: int) -> int:
        """Return the precise nanosecond timestamp for a single frame."""
        if self._all_timestamps is not None:
            return int(self._all_timestamps[idx])
        return int(self.get_metadata_arrays([_UNIX_T_NS], indices=[idx])[_UNIX_T_NS][0])

    def timestamps_at(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """
        Return precise nanosecond timestamps for multiple frames in one vectorized call.

        Much faster than calling ``timestamp_at`` in a loop — uses a single
        ``np.frombuffer`` pass per file group rather than per-frame Python byte reads.

        Parameters
        ----------
        indices:
            Frame indices to look up (any order; results match input order).

        Returns
        -------
        np.ndarray of int64, shape ``(len(indices),)``.
        """
        return self.get_metadata_arrays(
            [_UNIX_T_NS], indices=np.asarray(indices, dtype=np.int64)
        )[_UNIX_T_NS]

    def seek_time(self, timestamp_ns: int) -> int:
        """
        Return the index of the frame closest to ``timestamp_ns``.

        Uses a two-level binary search: first finds the file, then searches
        within it. Per-file timestamp arrays are cached after first access.
        """
        if self._total_frames == 0:
            return 0

        # Ensure per-file bounds are populated
        self._ensure_file_bounds()

        # Find the target file via coarse bounds
        file_idx = 0
        for i, bounds in enumerate(self._file_ts_bounds):
            if bounds is None:
                continue
            _first_ts, last_ts = bounds
            if timestamp_ns <= last_ts:
                file_idx = i
                break
            file_idx = i  # keep advancing to last file

        # Lazily build within-file timestamp array using the fast sequential path
        # (_composite_extract) rather than the indexed path to avoid Python-level loops.
        if self._file_timestamps[file_idx] is None:
            n = self._file_frame_counts[file_idx]
            mm = self._get_mmap(file_idx)
            ts_keys = self._timing_keys()
            if mm is not None:
                meta = self._composite_extract(mm, ts_keys, 0, n, self.frame_config)  # type: ignore[arg-type]
                self._file_timestamps[file_idx] = self._derive_unix_t_ns(meta, ts_keys)
            else:
                self._file_timestamps[file_idx] = np.zeros(n, dtype=np.int64)

        file_ts = self._file_timestamps[file_idx]
        assert file_ts is not None
        file_start = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0

        pos = bisect.bisect_left(file_ts, timestamp_ns)
        if pos == 0:
            return file_start
        if pos >= len(file_ts):
            return file_start + len(file_ts) - 1
        # Return the closer of the two neighbors
        if timestamp_ns - file_ts[pos - 1] <= file_ts[pos] - timestamp_ns:
            return file_start + pos - 1
        return file_start + pos

    def _ensure_file_bounds(self) -> None:
        """Eagerly read first+last frame timestamp for each file if not yet done."""
        ts_keys = self._timing_keys()
        for i, (n_frames, bounds) in enumerate(
            zip(self._file_frame_counts, self._file_ts_bounds, strict=False)
        ):
            if bounds is not None or n_frames == 0:
                continue
            file_start = self._cumulative_frames[i - 1] if i > 0 else 0
            probe_idx = np.array([file_start, file_start + n_frames - 1], dtype=np.int64)
            if n_frames == 1:
                probe_idx = np.array([file_start], dtype=np.int64)
            ts_vals = self.get_metadata_arrays([_UNIX_T_NS], indices=probe_idx)[_UNIX_T_NS]
            self._file_ts_bounds[i] = (int(ts_vals[0]), int(ts_vals[-1]))

    # ── diagnostics ───────────────────────────────────────────

    def print_metadata_offsets(self) -> None:
        print(f"Metadata Byte Offsets for {self.name}:")
        if not self.metadata_offsets:
            print("  (None found)")
            return
        for key, (start, end) in sorted(self.metadata_offsets.items()):
            print(f"  {key}: bytes {start}-{end} (width {end - start})")

    def verify_metadata_offsets(self, num_frames: int = 100) -> bool:
        """
        Verify vectorized metadata extraction against naive JSON parsing.

        Raises ``RuntimeError`` on empty/unconfigured sequences.
        Returns ``True`` if all checked frames match, ``False`` on first mismatch.
        """
        if self._total_frames == 0 or not self.frame_config or not self.metadata_offsets:
            raise RuntimeError("Cannot verify: sequence is empty or not configured.")
        check = min(num_frames, self._total_frames)
        keys = list(self.metadata_offsets.keys())
        vectorized = self.get_metadata_arrays(keys)
        conf = self.frame_config
        for i in range(check):
            file_idx, local_idx = self._locate_frame(i)
            mm = self._get_mmap(file_idx)
            if mm is None:
                continue
            offset = local_idx * conf.frame_size
            h = orjson.loads(mm[offset : offset + conf.header_size])
            for key in keys:
                if "." in key:
                    parent, child = key.split(".", 1)
                    naive = h.get(parent, {}).get(child, 0)
                else:
                    naive = h.get(key, 0)
                if int(naive) != int(vectorized[key][i]):
                    print(f"Mismatch at frame {i}, key '{key}': naive={naive}, vec={vectorized[key][i]}")
                    return False
        return True


# ─────────────────────────────────────────────────────────────
#  PanosetiRun
# ─────────────────────────────────────────────────────────────

class PanosetiRun:
    """
    Unified entrypoint for a PanoSETI observing run (.pffd directory).

    Provides lazy-loaded access to configs, metadata logs, housekeeping
    telemetry, and data products.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.products: dict[str, PFFSequence] = {}

        self._configs: dict[str, Any] | None = None
        self._metadata: dict[str, list[dict[str, Any]]] | None = None
        self._hk_data: dict[str, dict[str, np.ndarray]] | None = None
        self._manifests: dict[str, str] | None = None

        self._scan_products()

    def _scan_products(self) -> None:
        files_map: dict[str, list[Path]] = {}
        for f in self.run_dir.glob("*.pff"):
            if f.name == "hk.pff" or f.stat().st_size == 0:
                continue
            key = ".".join(
                p for p in f.name.split(".")
                if not (p.startswith("start") or p.startswith("seqno") or p == "pff")
            )
            files_map.setdefault(key, []).append(f)

        for k, v in files_map.items():
            try:
                seq = PFFSequence(v)
                if len(seq) > 0:
                    self.products[k] = seq
            except (OSError, ValueError) as e:
                logger.warning("Skipping product %s: %s", k, e)

    # ── data products ─────────────────────────────────────────

    def list_products(self) -> list[str]:
        return sorted(self.products.keys())

    def get_product(self, product_name: str) -> PFFSequence:
        if product_name not in self.products:
            raise KeyError(f"Product '{product_name}' not found. Available: {self.list_products()}")
        return self.products[product_name]

    # ── lazy properties ───────────────────────────────────────

    @property
    def configs(self) -> dict[str, Any]:
        if self._configs is None:
            self._load_configs()
        return self._configs  # type: ignore[return-value]

    @property
    def metadata(self) -> dict[str, list[dict[str, Any]]]:
        if self._metadata is None:
            self._load_metadata()
        return self._metadata  # type: ignore[return-value]

    @property
    def manifests(self) -> dict[str, str]:
        if self._manifests is None:
            self._load_manifests()
        return self._manifests  # type: ignore[return-value]

    # ── configs ───────────────────────────────────────────────

    def _load_configs(self) -> None:
        from .models import (
            DaqConfig,
            DataConfig,
            FirmwareConfig,
            NetworkConfig,
            ObsConfig,
            PhBaselineConfig,
            QuaboConfig,
            QuaboPhBaseline,
            QuaboUids,
        )

        models_map = {
            "data_config": DataConfig,
            "obs_config": ObsConfig,
            "daq_config": DaqConfig,
            "quabo_ph_baseline": PhBaselineConfig,
            "ph_baseline_config": PhBaselineConfig,
            "network_config": NetworkConfig,
            "quabo_uids": QuaboUids,
            "firmware_config": FirmwareConfig,
        }

        self._configs = {}
        for f in self.run_dir.glob("*.json"):
            if f.name.startswith("dp_manifest"):
                continue
            try:
                data = orjson.loads(f.read_bytes())
                model = models_map.get(f.stem) or (
                    QuaboConfig if f.stem.startswith("quabo_config") else None
                )
                self._configs[f.stem] = model(**data) if model else data
            except (orjson.JSONDecodeError, ValidationError, OSError) as e:
                logger.warning("Skipping config %s: %s", f.name, e)

        for f in self.run_dir.glob("*.toml"):
            try:
                import tomllib
                self._configs[f.stem] = tomllib.loads(f.read_text())
            except (OSError, Exception) as e:
                logger.warning("Skipping toml %s: %s", f.name, e)

    def get_config(self, name: str) -> Any:
        clean = name.replace(".json", "").replace(".toml", "")
        return self.configs.get(clean)

    # ── JSONL metadata ────────────────────────────────────────

    def _load_metadata(self) -> None:
        self._metadata = {}
        for f in self.run_dir.glob("*.jsonl"):
            try:
                lines = f.read_text().splitlines()
                self._metadata[f.name] = [
                    orjson.loads(line) for line in lines if line.strip()
                ]
            except (orjson.JSONDecodeError, OSError) as e:
                logger.warning("Skipping jsonl %s: %s", f.name, e)

    def get_metadata_log(self, name: str) -> list[dict[str, Any]]:
        return self.metadata.get(name, [])

    # ── manifests ─────────────────────────────────────────────

    def _load_manifests(self) -> None:
        self._manifests = {}
        for f in self.run_dir.glob("dp_manifest*"):
            try:
                self._manifests[f.name] = f.read_text()
            except OSError as e:
                logger.warning("Skipping manifest %s: %s", f.name, e)

    def get_manifest(self, name: str) -> str:
        return self.manifests.get(name, "")

    # ── logs ──────────────────────────────────────────────────

    def list_logs(self) -> list[str]:
        return sorted(
            f.name
            for ext in ["*.log", "*.txt"]
            for f in self.run_dir.glob(ext)
            if not f.name.startswith("dp_manifest")
        )

    def get_log(self, name: str) -> str:
        log_path = self.run_dir / name
        if not log_path.exists():
            raise FileNotFoundError(f"Log '{name}' not found in {self.run_dir}")
        return log_path.read_text()

    # ── housekeeping ──────────────────────────────────────────

    def get_hk(self) -> dict[str, dict[str, np.ndarray]]:
        """Lazy-load and parse hk.pff into {device: {field: np.ndarray}}."""
        if self._hk_data is not None:
            return self._hk_data
        self._hk_data = _parse_hk_file(self.run_dir / "hk.pff")
        return self._hk_data

    # ── display ───────────────────────────────────────────────

    def show(self, details: bool = False) -> None:
        from rich.console import Console
        from rich.tree import Tree

        console = Console()
        tree = Tree(f"[bold gold1]Run: {self.run_dir.resolve().name}[/]")

        _ = self.configs, self.metadata, self.manifests

        if self._configs or self._metadata:
            cb = tree.add("Configurations & Metadata")
            for k in sorted(self._configs or {}):
                cb.add(f"[cyan]{k}[/]")
            for k in sorted(self._metadata or {}):
                cb.add(f"[blue]{k}[/]")

        if self._manifests:
            mb = tree.add("Data Manifests")
            for k in sorted(self._manifests):
                mb.add(f"[green]{k}[/]")

        logs = self.list_logs()
        if logs:
            lb = tree.add("Logs")
            for lg in logs:
                lb.add(f"[magenta]{lg}[/]")

        if self.products:
            pb = tree.add("Data Products")
            for name, seq in sorted(self.products.items()):
                b = pb.add(f"[bold green]{name}[/] ({len(seq):,} frames)")
                if details:
                    for fp in seq.file_paths:
                        b.add(f"[dim]{fp.name} ({fp.stat().st_size / 1024**2:.1f} MB)[/]")

        hk = self.run_dir / "hk.pff"
        if hk.exists():
            tree.add(f"[bold white]Housekeeping:[/] hk.pff ({hk.stat().st_size / 1024:.1f} KB)")

        console.print(tree)


# ─────────────────────────────────────────────────────────────
#  Legacy Compatibility Shims
# ─────────────────────────────────────────────────────────────

class hkpff:
    """Legacy shim for reading hk.pff files. Prefer ``PanosetiRun.get_hk()``."""

    def __init__(self, fn: str | Path = "hk.pff") -> None:
        self.fn = Path(fn)
        self.hk_info: dict[str, dict[str, np.ndarray]] = {}

    def readhk(self) -> dict[str, dict[str, np.ndarray]]:
        self.hk_info = _parse_hk_file(self.fn)
        return self.hk_info


class qconfig:
    """Legacy shim for loading configuration JSON files. Prefer ``PanosetiRun.get_config()``."""

    def __init__(self, fn: str) -> None:
        from glob import glob
        self.config: dict[str, Any] = {}
        jfiles = glob(fn)
        if not jfiles:
            raise FileNotFoundError(f"No config files matched: {fn!r}")

        from .models import QuaboConfig
        for file in jfiles:
            p = Path(file)
            try:
                data = orjson.loads(p.read_bytes())
                if p.stem.startswith("quabo_config"):
                    self.config[p.stem] = QuaboConfig(**data).model_dump()
                else:
                    self.config[p.stem] = data
            except (orjson.JSONDecodeError, ValidationError, OSError) as e:
                logger.warning("Skipping config %s: %s", p.name, e)
