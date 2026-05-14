# PanoSETI PFF → Zarr v3 Specification

**Version**: 1.0  
**Status**: Active  
**`panoseti_pff_zarr_version`**: `"1.0"`

This document defines the Zarr v3 storage layout produced by `pypff.zarr.convert_run`
and read by `pypff.zarr.PanosetiZarrRun` / `PanosetiZarrStore`.

---

## 1. Overview

One Zarr v3 store is produced per `(data_product, module)` pair in a `.pffd`
observation run.  The store is a self-contained artifact: it holds pixel data,
nanosecond-precision timestamps, per-frame header fields, and embedded run
configuration.  A sibling `.panoseti-meta/` directory holds ancillary run files
(logs, raw configs, housekeeping).

---

## 2. Store naming

```
<out_dir>/
<run_dir_name>.<data_product>.<module>.zarr     ← per-product store
<run_dir_name>.panoseti-meta/                   ← run-level sidecar bundle
```

`<run_dir_name>` is the `.pffd` directory name (e.g.
`obs_Lick.start_2024-07-25T04:34:06Z.runtype_sci-data.pffd`).
`<data_product>` and `<module>` are derived from the PFF filename convention
(`dp_ph256.bpp_2.module_254`, etc.).

---

## 3. Group structure — single flat root group

**All arrays live at the root of the Zarr store** (no sub-groups).

Rationale (empirically validated, 2026-05-13, xarray 2026.4.0 + zarr-python 3.2.1):

| Access pattern | Hierarchical (`headers/pkt_num`) | **Flat (`pkt_num`)** |
|---|---|---|
| `xr.open_zarr(store)` — the dominant idiom | Headers invisible | **All variables auto-discovered** |
| `xr.open_datatree(store)` | Works but `.to_dataset()` ≠ flat; manual merge required | Single Dataset, no merge needed |
| TensorStore / Rust zarrs / Julia Zarr.jl | Sub-group traversal required | Root key enumeration — universal |
| Spec compliance | Relies on consolidated metadata (non-spec for Zarr v3) | Fully spec-compliant |

Logical grouping of headers is expressed via:
- `header_fields` root attribute (list of single-level header array names)
- `quabo_fields` root attribute (list of module-level header array names)
- Per-array `pff_header_field` attribute (original dotted key, e.g. `"quabo_0.pkt_num"`)

---

## 4. Required arrays

| Array | dtype | shape | `_ARRAY_DIMENSIONS` | Description |
|---|---|---|---|---|
| `images` | `FrameConfig.dtype` (`int16`, `uint16`, `uint8`) | `(T, H, W)` | `["time", "y", "x"]` | Raw pixel data, frame-major |
| `unix_t_ns` | `int64` | `(T,)` | `["time"]` | UNIX timestamp in nanoseconds — the authoritative time axis |

`unix_t_ns` is resolved from hardware clocks via `pypff.utils.get_precise_time_ns`
and stored once.  It is not losslessly derivable from the raw header fields if
WR/DAQ clocks desync, so it is kept as an independent canonical column.

---

## 5. Optional arrays — header fields

### 5a. Single-level headers (ph256 data product)

| Array | dtype | shape | Source PFF field |
|---|---|---|---|
| `pkt_num` | `uint32` | `(T,)` | `pkt_num` |
| `pkt_tai` | `uint16` | `(T,)` | `pkt_tai` (10-bit WR counter) |
| `pkt_nsec` | `uint32` | `(T,)` | `pkt_nsec` |
| `tv_sec` | `int64` | `(T,)` | `tv_sec` |
| `tv_usec` | `uint32` | `(T,)` | `tv_usec` |
| `quabo_num` | `uint8` | `(T,)` | `quabo_num` (0–3) |

### 5b. Module-level headers (img8, img16, ph1024 data products)

For `i ∈ {0, 1, 2, 3}`:

| Array | dtype | shape | Source PFF field |
|---|---|---|---|
| `quabo_<i>_pkt_num` | `uint32` | `(T,)` | `quabo_<i>.pkt_num` |
| `quabo_<i>_pkt_tai` | `uint16` | `(T,)` | `quabo_<i>.pkt_tai` |
| `quabo_<i>_pkt_nsec` | `uint32` | `(T,)` | `quabo_<i>.pkt_nsec` |
| `quabo_<i>_tv_sec` | `int64` | `(T,)` | `quabo_<i>.tv_sec` |
| `quabo_<i>_tv_usec` | `uint32` | `(T,)` | `quabo_<i>.tv_usec` |

### Dtype rationale

| Field | dtype | Reason |
|---|---|---|
| `quabo_num` | `uint8` | ∈ {0,1,2,3} — 2 bits used |
| `pkt_tai` | `uint16` | 10-bit WR counter (0–1023) |
| `pkt_num` | `uint32` | Per-file packet counter; file ≤ 1 GB → ≤ ~10⁷ frames |
| `pkt_nsec` | `uint32` | ∈ [0, 10⁹) |
| `tv_usec` | `uint32` | ∈ [0, 10⁶) |
| `tv_sec` | `int64` | Unix seconds — int64 for future-safety |
| `unix_t_ns` | `int64` | ns since epoch — 19 significant digits, requires int64 |

Compared to storing all fields as `int64`: single-level headers shrink from
48 B/frame → 16 B/frame; module-level from 160 B/frame → 52 B/frame.  After
zstd on the resulting monotonic columns, effective overhead is ~2–5 B/frame.

---

## 6. Required root attributes

The root group's `zarr.json` MUST contain:

| Attribute | Type | Description |
|---|---|---|
| `panoseti_pff_zarr_version` | `str` | Spec version — `"1.0"` |
| `data_product` | `str` | Data product type (e.g. `"ph256"`, `"img16"`) |
| `bytes_per_pixel` | `int` | Bytes per pixel |
| `module` | `str` | Module identifier |
| `source_pff_files` | `list[str]` | Filenames of source `.pff` files (not full paths) |
| `total_frames` | `int` | Total frame count across all source files |
| `header_format` | `str` | `"single"` (ph256) or `"module"` (img/ph1024) |
| `header_fields` | `list[str]` | Root-level single-level header array names |
| `quabo_fields` | `list[str]` | Root-level module-level header array names |
| `frame_config` | `dict` | `{header_size, payload_size, frame_size, image_shape, dtype_str, bytes_per_pixel, format_name}` |

Optional root attributes:

| Attribute | Type | Description |
|---|---|---|
| `run_configs` | `dict` | Parsed run configs embedded at conversion time |

---

## 7. Per-array attributes

All arrays carry:

| Attribute | Description |
|---|---|
| `_ARRAY_DIMENSIONS` | List of dimension names (zarr v2 xarray compat) |

`unix_t_ns` additionally carries:

| Attribute | Value | Description |
|---|---|---|
| `ns_since_epoch` | `"1970-01-01T00:00:00Z"` | Non-CF units key — prevents xarray from attempting CF time-decoding (which would fail because `"ns"` is not a valid CF time unit) |
| `long_name` | `"UNIX timestamp in nanoseconds"` | Human-readable description |

Header arrays additionally carry:

| Attribute | Example | Description |
|---|---|---|
| `pff_header_field` | `"quabo_0.pkt_num"` | Original dotted key from PFF header schema, for provenance |

---

## 8. Codec chain

Default: `bytes → zstd(level=3)` (via zarr-python `ZstdCodec`).

| Codec | `convert_run` parameter |
|---|---|
| zstd (default) | `codec="zstd", level=3` |
| Blosc-LZ4 | `codec="blosc-lz4", level=3` |
| gzip | `codec="gzip", level=3` |
| none | `codec="none"` |

Consolidated metadata (`zarr.consolidate_metadata`) is **not written** by default.
It is flagged as non-spec for Zarr v3 by zarr-python 3.2+ and may break
non-zarr-python readers.  Enable with `ZarrPythonWriter(consolidate=True)` if
you are reading from S3 and open-time latency is measurable.

---

## 9. Chunk shape rules

### Image arrays

Chunks span time only: `(C, H, W)` where C is auto-sized to ~8 MB pre-compression:

| Data product | Frame bytes | Auto C | Chunk size |
|---|---|---|---|
| `ph256` (16×16 int16) | 512 B | 16384 | 8 MB |
| `img16` (32×32 int16) | 2 KB | 4096 | 8 MB |
| `ph1024` (32×32 int16) | 2 KB | 4096 | 8 MB |
| `img8` (32×32 uint8) | 1 KB | 8192 | 8 MB |

Override via `convert_run(time_chunk=N)`.

### 1-D arrays (timestamps, headers)

Chunk size = `C × 2`, so each image chunk aligns with half a header/timestamp
chunk for efficient joint time-range reads.

---

## 10. Run bundle layout

When `convert_run(write_sidecars=True)` (default), a sibling sidecar bundle is
written to preserve all non-PFF ancillary files from the `.pffd` directory:

```
<out_dir>/
<run>.panoseti-meta/
  manifest.json          ← machine-readable index (version, timestamps, store list)
  configs/               ← raw .json and .toml files copied verbatim
    obs_config.json
    daq_config.json
    data_config.json
    quabo_config_*.json
    sw_info.json
    …
  logs/                  ← log.txt, hp_stdout_*, *.log
  manifests/             ← dp_manifest*.json / *.jsonl
  sentinels/             ← collect_complete, run_complete, recording_ended
  hk.pff                 ← housekeeping data (PFF format, preserved verbatim)
```

`PanosetiZarrRun` reads configs from `configs/` first; if absent it falls back
to `run_configs` embedded in any one store's root attributes.

---

## 11. Compatibility matrix

| Reader | Minimum version | Access pattern | Notes |
|---|---|---|---|
| zarr-python | ≥ 3.0 | `zarr.open_group(store, mode="r")["images"][:]` | Native; recommended for raw ndarray access |
| xarray | ≥ 2024.1 | `xr.open_zarr(store, consolidated=False)` | All arrays auto-discovered as Dataset variables |
| xarray DataTree | ≥ 2024.10 | `xr.open_datatree(store, engine="zarr")` | Works but unnecessary — flat layout means root Dataset already contains everything |
| dask | ≥ 2024.1 | `da.from_zarr(store, component="images")` | Lazy, parallelizable chunk reads |
| TensorStore | ≥ 0.3 | `ts.open({"driver": "zarr3", "kvstore": str(store)})` | Zarr v3 spec-compliant; no sub-groups to traverse |
| Julia Zarr.jl | ≥ 0.8 | `zopen(store; fill_as_missing=false)["images"]` | Standard zarr v3; flat root enumeration |
| Rust zarrs | ≥ 0.15 | `zarrs::storage::open_group(...)` | Spec-compliant |

---

## 12. Versioning policy

`panoseti_pff_zarr_version` follows semantic versioning applied to the store
schema:

- **Patch** (1.0.x): backwards-compatible additions (new optional arrays or attrs).
- **Minor** (1.x.0): new required arrays or attrs; old readers still open stores safely.
- **Major** (x.0.0): breaking layout change.  Readers MUST refuse to open stores whose major version exceeds their supported major version.

Version `"1.0"` (this document) is the initial production release.

---

## 13. Version history

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-05-13 | Initial release. Flat root layout; `header_fields` / `quabo_fields` discoverability; `run_configs` embedding; `.panoseti-meta/` sidecar bundle. |
