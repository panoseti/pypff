# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pypff` is a Python I/O library for PanoSETI File Format (PFF) data files — binary science data produced by the PanoSETI telescope array. It provides zero-copy mmap-based access, Pydantic-validated headers, and a high-level run discovery API.

## Commands

All commands assume `uv sync` has been run first.

```bash
uv sync                          # Install all dependencies
uv run pypff test all            # Run full test suite (unit + logic + legacy)
uv run pypff test unit           # Tier 1 unit tests only
uv run pypff test logic          # Tier 2 logic/IO tests only
uv run pypff test legacy         # Legacy integration tests only
uv run pypff test all --lint     # Also run Ruff + MyPy
uv run pypff test all --cov      # With coverage reporting
```

Run a single test file directly:
```bash
uv run pytest src/ci/tier1_unit/test_utils.py
uv run pytest src/ci/tier2_logic/test_io.py::test_name
```

Lint manually:
```bash
uv run ruff check .
uv run mypy src
```

## Architecture

### Two-Layer I/O Design

**Legacy layer (`src/pypff/io.py`):** Original byte-offset based reader. Uses hardcoded byte positions (`loc_arr`) to extract metadata fields from fixed-format PFF JSON headers. Exposes `datapff`, `hkpff`, `qconfig`. Kept for backward compatibility.

**Modern layer (`src/pypff/io2.py`):** `PFFSequence` and `PanosetiRun` — the primary API. `PFFSequence` wraps one or more sequential `.pff` files for a single data product, providing:
- `mmap`-based zero-copy frame access
- Standard Python slicing (`seq[0:100:10]`) returning NumPy arrays
- Pickle-safe multiprocessing (file handles dropped on `__getstate__`, lazily reopened)
- Nanosecond-precision `seek_time()` for timestamp-based navigation
- `get_all_metadata()` for bulk header extraction

`PanosetiRun` is a lazy-loaded directory scanner over a `.pffd` run directory. It discovers and groups `.pff` files by data product and exposes typed config loading via Pydantic models.

### Models (`src/pypff/models.py`)

Pydantic v2 models validate all PFF headers and PANOSETI config JSON files. `BaseStrictModel` (extra=`'forbid'`) is the base for all models to catch config key typos. Key models: `PFFHeader`, `QuaboHeader`, `ModuleHeader`, `FrameConfig`.

### Timing

All timestamps are **nanosecond integers** (not floats) to avoid floating-point precision loss. `utils.get_precise_time_ns()` reconciles `tv_sec`/`tv_usec` (system clock) against `pkt_nsec`/`pkt_tai` (quabo hardware clock) using 10-bit TAI counter wraparound logic.

### Pixel Maps

`src/pypff/pixelmap.py` and the four `pixelmap_*.py` files provide MAROC↔physical pixel coordinate conversions for BGA and QFP package variants. These are static lookup tables.

### CLI (`src/pypff/cli.py`, `src/pypff/_cli/`)

Built with `typer`. Entry point is `pypff` (defined in `pyproject.toml`). Sub-commands live in `src/pypff/_cli/`. The `test` sub-command (`_cli/test.py`) delegates to `pytest` via `subprocess`.

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
