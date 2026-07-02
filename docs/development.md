# pypff Development Guide

## Setup

```bash
uv sync                     # install all dependencies (+ dev extras)
uv sync --extra zarr        # also install zarr-python, xarray, dask
```

## Commands

```bash
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

## Test tiers

| Tier | Path | What it covers |
|------|------|----------------|
| 1 — unit | `src/ci/tier1_unit/` | Models, utils — fast, no file I/O |
| 2 — logic | `src/ci/tier2_logic/` | IO correctness, slicing, timestamps, concurrency |
| legacy | `src/ci/legacy_tests/` | Original test suite against sample `.pff` files |

### Key tier-2 test files

| File | Coverage |
|------|----------|
| `test_io2_streaming.py` | `iter_batches`, `iter_byte_range`, `__getitem__`, `timestamps`, `timestamp_at`, **`timestamps_at`**, `get_metadata_arrays` (sequential + indexed paths), LRU, context manager |
| `test_io2_new_features.py` | Slicing, multiprocessing, `seek_time`, **`timestamps_at`** on synthetic multi-file data |
| `test_io2_vectorization.py` | Module-header metadata structuring, housekeeping parsing, multiprocessing |
| `test_io2_comprehensive.py` | End-to-end `PanosetiRun` + `PFFSequence` against real example data |
| `test_io_comparison.py` | `io.py` vs `io2.py` parity |

Test data for legacy tests lives alongside the tests in
`src/ci/legacy_tests/{hk-data,sci-data,config-data}/`.

## Zarr CLI

```bash
uv run pypff zarr <obs.pffd> <out_dir>   # convert a PFF run to Zarr v3
```

See [`docs/zarr_v3_spec.md`](zarr_v3_spec.md) for the full L0 store layout specification.
