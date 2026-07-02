# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pypff` is a Python I/O library for PanoSETI File Format (PFF) data files — binary science data produced by the PanoSETI telescope array. It provides zero-copy mmap-based access, Pydantic-validated headers, and a high-level run discovery API.

## Commands

_See [`docs/development.md`](docs/development.md) for setup, test, and lint commands._

## Architecture

### Two-Layer I/O Design

**Legacy layer (`src/pypff/io.py`):** Original byte-offset based reader. Uses hardcoded byte positions (`loc_arr`) to extract metadata fields from fixed-format PFF JSON headers. Exposes `datapff`, `hkpff`, `qconfig`. Kept for backward compatibility.

**Modern layer (`src/pypff/io2.py`):** `PFFSequence` and `PanosetiRun` — the primary API. `PFFSequence` wraps one or more sequential `.pff` files for a single data product, providing:
- Streaming-by-default API: `for img in seq` and `for batch in seq.iter_batches(size=256)` are zero-copy within a single file.
- `iter_byte_range(file_idx, byte_start, byte_end, batch_size)` for distributed chunked reads (Nextflow/Dask workers).
- `read_images(indices)` for sorted-with-inverse-permutation random access; `read_images_range(start, count)` for sequential bulk.
- `seq[i]` returns a zero-copy view; `seq[start:stop:step]` uses `read_images`.
- Single-pass metadata extraction via NumPy composite structured dtype (`get_metadata_arrays`). Supports virtual `unix_t_ns` key.
- `timestamps(indices=None)` returns cached `int64` ns array; `timestamps(as_datetime=True)` returns a zero-copy `datetime64[ns]` view for matplotlib/pandas. `timestamp_at(i)` and `seek_time(ns)` use a two-level binary search (file bounds, then within-file).
- LRU-bounded mmap handles (`_MmapLRU`, default capacity 16). `PFFSequence` is a context manager.
- Pickle-safe for multiprocessing (handles dropped on `__getstate__`, lazily reopened).
- `get_frame(i)` returns `(dict, ndarray_view)` by default; `get_frame_validated(i)` returns `(QuaboHeader|ModuleHeader, ndarray_view)`.

`PanosetiRun` is a lazy-loaded directory scanner over a `.pffd` run directory. It discovers and groups `.pff` files by data product and exposes typed config loading via Pydantic models.

### Models (`src/pypff/models.py`)

Pydantic v2 models validate all PFF headers and PANOSETI config JSON files. `BaseStrictModel` (extra=`'forbid'`) is the base for all models to catch config key typos. Key models: `PFFHeader`, `QuaboHeader`, `ModuleHeader`, `FrameConfig`.

### Timing

All timestamps are **`int64` nanoseconds since the Unix epoch** — never Python floats. Float64 has only ~15–16 significant digits; a Unix timestamp in nanoseconds is ~19 digits, so `float(ns) / 1e9` loses nanosecond precision at the point of division.

- `timestamps()` → `np.ndarray[int64]` — for arithmetic, diffs, and storage.
- `timestamps(as_datetime=True)` → `np.ndarray[datetime64[ns]]` — zero-copy view of the same `int64` data; natively understood by matplotlib date axes, pandas, and xarray. Use for display only.
- `timestamp_at(i)` → `int` — single frame, nanoseconds.

**Integer-space epoch rule:** when converting to float seconds for plotting, subtract the reference epoch first in integer space, then divide — the resulting relative values are small so float64 retains sub-nanosecond resolution:
```python
# CORRECT — subtract first (int64), then divide (small values, no precision loss)
rel_s = (times_ns - t0_ns) / 1e9

# WRONG — 1.7e18 ÷ 1e9 ≈ 1.7e9 s, only ~6 decimal digits remain after the decimal
times_ns / 1e9 - t0_s
```

`utils.get_precise_time_ns()` reconciles `tv_sec`/`tv_usec` (system clock) against `pkt_nsec`/`pkt_tai` (quabo hardware clock) using 10-bit TAI counter wraparound logic.

### Pixel Maps

`src/pypff/pixelmap.py` and the four `pixelmap_*.py` files provide MAROC↔physical pixel coordinate conversions for BGA and QFP package variants. These are static lookup tables.

### Zarr v3 Conversion (`src/pypff/zarr/`, optional extra `pypff[zarr]`)

Install with `uv sync --extra zarr`. The zarr extra requires zarr-python ≥ 3, xarray, and dask.

**Conversion:**
```bash
uv run pypff zarr <obs.pffd> <out_dir>   # CLI
```
```python
from pypff import PanosetiRun
from pypff.zarr import convert_run
stores = convert_run(PanosetiRun("obs.pffd"), "out/")
```

**Reading:**
```python
from pypff.zarr import PanosetiZarrRun
zrun = PanosetiZarrRun("out/")
store = zrun.get_product("dp_ph256.bpp_2.module_254")
store.timestamps()    # int64 ns
store.to_dataset()    # xarray.Dataset — images + unix_t_ns + all header fields
```

**Key design decisions (see [`docs/zarr_v3_spec.md`](docs/zarr_v3_spec.md) for the full spec):**
- **Flat root layout**: all arrays (`images`, `unix_t_ns`, header fields) live at the zarr store root — no sub-groups. `xr.open_zarr(store)` surfaces every variable automatically. Sub-groups are invisible to `xr.open_zarr` without `group=`, so the hierarchical layout was rejected.
- **Module-level header naming**: `quabo_0.pkt_num` → `quabo_0_pkt_num` (dots replaced by underscores) for xarray compatibility.
- **Discoverability attrs**: `header_fields` and `quabo_fields` lists in root attrs let consumers separate image columns from metadata columns without re-deriving naming conventions.
- **No consolidated metadata**: `zarr.consolidate_metadata()` is not called by default — it is non-spec for Zarr v3 and emits `ZarrUserWarning`. Use `xr.open_zarr(store, consolidated=False)`.
- **Sidecar bundle**: `convert_run` writes a sibling `<run>.panoseti-meta/` directory with all non-PFF ancillary files (configs/, logs/, hk.pff, sentinels/). Run configs are also embedded in each store's root attrs under `run_configs`.

**Key files:**
- `src/pypff/zarr/__init__.py` — `ZarrWriter` protocol, `ZarrPythonWriter`, `PFFToZarrConverter`, `convert_run`
- `src/pypff/zarr/_reader.py` — `PanosetiZarrStore`, `PanosetiZarrRun`, `open_zarr_run`
- `docs/zarr_v3_spec.md` — full store layout specification

### CLI (`src/pypff/cli.py`, `src/pypff/_cli/`)

Built with `typer`. Entry point is `pypff` (defined in `pyproject.toml`). Sub-commands live in `src/pypff/_cli/`. The `test` sub-command (`_cli/test.py`) delegates to `pytest` via `subprocess`. The `zarr` sub-command (`_cli/zarr.py`) wraps `convert_run`.

### Test Structure

```
src/ci/
  tier1_unit/      # Fast unit tests (models, utils)
  tier2_logic/     # IO correctness, slicing, concurrency
  legacy_tests/    # Original test suite against sample .pff files
```

Test data for legacy tests lives alongside the tests in `src/ci/legacy_tests/{hk-data,sci-data,config-data}/`.

## Key Conventions

- `ruff` is the linter/formatter; `mypy` runs in strict mode with pydantic plugin.
- Line length limit is 100 (ruff), but E501 is ignored so mypy drives strictness.
- `orjson` (not `json`) is used for all JSON parsing in the hot path — it's faster and handles bytes directly.
- Data product names follow the pattern `dp_<type>.bpp_<bits>.module_<id>`, e.g. `dp_img16.bpp_2.module_1`.
