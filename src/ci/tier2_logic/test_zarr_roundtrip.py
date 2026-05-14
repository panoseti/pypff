"""
Zarr v3 roundtrip tests for pypff.zarr.

Verifies:
- images round-trip bitwise through PFF → Zarr → numpy
- unix_t_ns matches PFFSequence.timestamps() exactly
- header column values match get_metadata_arrays
- header dtypes are the shrunk versions (uint8/uint16/uint32)
- module-level (img16) headers nest under headers/quabo_<i>/
- ZarrWriter protocol is satisfied by a no-op implementation
- xarray.open_zarr opens the store cleanly (if xarray installed)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="zarr not installed; run: uv sync --extra zarr")


# ── fixture helpers ──────────────────────────────────────────────────────────

def _make_ph256_pff(path: Path, n_frames: int = 20, start_idx: int = 0) -> None:
    """Single-level (ph256) PFF: 16x16 int16 payload, quabo_num header."""
    header_base = (
        '{"quabo_num":          0, "pkt_num": %10d, "pkt_tai":          0, '
        '"pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec":          0}\n\n*'
    )
    payload = np.arange(256, dtype=np.int16).reshape(16, 16).tobytes()
    with open(path, "wb") as f:
        for i in range(start_idx, start_idx + n_frames):
            h = header_base % (i, i * 1000)
            f.write(h.encode() + payload)


def _make_img16_pff(path: Path, n_frames: int = 15, start_idx: int = 0) -> None:
    """Module-level (img16) PFF: 32x32 uint16 payload, quabo_0..3 header."""
    header_base = (
        '{"quabo_0": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, '
        '"tv_sec": 1700000000, "tv_usec":          0}, '
        '"quabo_1": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, '
        '"tv_sec": 1700000000, "tv_usec":          0}, '
        '"quabo_2": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, '
        '"tv_sec": 1700000000, "tv_usec":          0}, '
        '"quabo_3": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, '
        '"tv_sec": 1700000000, "tv_usec":          0}}\n\n*'
    )
    payload = np.arange(1024, dtype=np.uint16).reshape(32, 32).tobytes()
    with open(path, "wb") as f:
        for i in range(start_idx, start_idx + n_frames):
            h = header_base % (
                i, 0, i * 1000,
                i, 0, i * 1000,
                i, 0, i * 1000,
                i, 0, i * 1000,
            )
            f.write(h.encode() + payload)


@pytest.fixture
def ph256_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "ph256_test.pffd"
    run_dir.mkdir()
    _make_ph256_pff(
        run_dir / "start_2024-01-01T00:00:00Z.dp_ph256.bpp_2.module_1.seqno_0.pff",
        n_frames=20, start_idx=0,
    )
    _make_ph256_pff(
        run_dir / "start_2024-01-01T00:00:10Z.dp_ph256.bpp_2.module_1.seqno_1.pff",
        n_frames=20, start_idx=20,
    )
    return run_dir


@pytest.fixture
def img16_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "img16_test.pffd"
    run_dir.mkdir()
    _make_img16_pff(
        run_dir / "start_2024-01-01T00:00:00Z.dp_img16.bpp_2.module_1.seqno_0.pff",
        n_frames=15, start_idx=0,
    )
    return run_dir


# ── roundtrip: images ─────────────────────────────────────────────────────────

def test_images_roundtrip_ph256(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")
    assert len(stores) == 1

    pff_seq = run.get_product(run.list_products()[0])
    expected = pff_seq.read_images_range(0)

    store = zarr.open_group(str(stores[0]), mode="r")
    np.testing.assert_array_equal(store["images"][:], expected)


def test_images_roundtrip_img16(img16_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(img16_run)
    stores = convert_run(run, tmp_path / "out")
    assert len(stores) == 1

    pff_seq = run.get_product(run.list_products()[0])
    expected = pff_seq.read_images_range(0)

    store = zarr.open_group(str(stores[0]), mode="r")
    np.testing.assert_array_equal(store["images"][:], expected)


# ── roundtrip: timestamps ─────────────────────────────────────────────────────

def test_timestamps_match_pff(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    pff_seq = run.get_product(run.list_products()[0])
    expected_ts = pff_seq.timestamps()

    store = zarr.open_group(str(stores[0]), mode="r")
    zarr_ts = store["unix_t_ns"][:]
    assert zarr_ts.dtype == np.dtype("int64")
    np.testing.assert_array_equal(zarr_ts, expected_ts)


# ── roundtrip: header values ─────────────────────────────────────────────────

def test_header_values_match_pff(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    pff_seq = run.get_product(run.list_products()[0])
    pff_meta = pff_seq.get_metadata_arrays(["pkt_num"])

    store = zarr.open_group(str(stores[0]), mode="r")
    # Headers are flat root-level arrays; xarray.open_zarr sees them as variables.
    zarr_pkt_num = store["pkt_num"][:].astype(np.int64)
    np.testing.assert_array_equal(zarr_pkt_num, pff_meta["pkt_num"])


# ── header dtypes ─────────────────────────────────────────────────────────────

def test_header_dtypes_single_level(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")

    expected = {
        "quabo_num": np.dtype("uint8"),
        "pkt_num":   np.dtype("uint32"),
        "pkt_tai":   np.dtype("uint16"),
        "pkt_nsec":  np.dtype("uint32"),
        "tv_sec":    np.dtype("int64"),
        "tv_usec":   np.dtype("uint32"),
    }
    for field, expected_dtype in expected.items():
        if field in store:
            assert store[field].dtype == expected_dtype, (
                f"{field}: got {store[field].dtype}, expected {expected_dtype}"
            )


def test_header_dtypes_module_level(img16_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(img16_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    # Module-level headers are flat: "quabo_0.pkt_num" → "quabo_0_pkt_num"
    assert store["quabo_0_pkt_num"].dtype == np.dtype("uint32")
    assert store["quabo_0_pkt_tai"].dtype == np.dtype("uint16")
    assert store["quabo_0_pkt_nsec"].dtype == np.dtype("uint32")
    assert store["quabo_0_tv_sec"].dtype == np.dtype("int64")
    assert store["quabo_0_tv_usec"].dtype == np.dtype("uint32")


# ── module-level header structure ─────────────────────────────────────────────

def test_module_header_flat_names(img16_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(img16_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")

    # All four quabo field sets present as flat root-level arrays
    for qi in range(4):
        assert f"quabo_{qi}_pkt_num" in store, f"quabo_{qi}_pkt_num missing"

    # quabo_num is single-level only; not present in module-level stores
    assert "quabo_num" not in store


# ── ZarrWriter protocol ───────────────────────────────────────────────────────

def test_no_op_writer_satisfies_protocol() -> None:
    from pypff.zarr import ZarrWriter

    class RecordingWriter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def create_store(self, path: Path | str) -> Any:
            self.calls.append("create_store")
            return {}

        def create_array(self, root_group: Any, name: str, shape: Any, chunks: Any, dtype: Any) -> Any:
            self.calls.append(f"create_array:{name}")
            return {}

        def write_slice(self, array: Any, slices: Any, data: Any) -> None:
            self.calls.append("write_slice")

        def set_attrs(self, obj: Any, attrs: dict[str, Any]) -> None:
            self.calls.append("set_attrs")

        def finalize(self, path: Path | str) -> None:
            self.calls.append("finalize")

    assert isinstance(RecordingWriter(), ZarrWriter)


# ── store attributes ──────────────────────────────────────────────────────────

def test_root_attrs_present(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    attrs = dict(store.attrs)
    assert "panoseti_pff_zarr_version" in attrs
    assert "frame_config" in attrs
    assert attrs["data_product"] == "ph256"
    assert attrs["total_frames"] == 40  # 2 files x 20 frames


def test_unix_t_ns_attrs(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    ts_attrs = dict(store["unix_t_ns"].attrs)
    assert "ns_since_epoch" in ts_attrs
    assert "_ARRAY_DIMENSIONS" in ts_attrs


# ── xarray integration ────────────────────────────────────────────────────────

def test_xarray_open(ph256_run: Path, tmp_path: Path) -> None:
    pytest.importorskip("xarray")
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    import xarray as xr
    ds = xr.open_zarr(str(stores[0]), consolidated=False)
    assert "images" in ds
    assert "unix_t_ns" in ds
    assert ds.images.dims == ("time", "y", "x")
    assert ds["unix_t_ns"].dtype == np.dtype("int64")
    # Header arrays must appear as dataset variables (flat root-level layout)
    assert "pkt_num" in ds
    assert "quabo_num" in ds
    assert ds["pkt_num"].dims == ("time",)
    assert ds["pkt_num"].dtype == np.dtype("uint32")


# ── discoverability attrs ─────────────────────────────────────────────────────

def test_header_fields_single_level(ph256_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    attrs = dict(zarr.open_group(str(stores[0]), mode="r").attrs)
    assert "header_fields" in attrs
    assert "quabo_fields" in attrs
    hf = set(attrs["header_fields"])
    assert hf == {"pkt_num", "pkt_tai", "pkt_nsec", "tv_sec", "tv_usec", "quabo_num"}
    assert attrs["quabo_fields"] == []  # ph256 has no per-quabo fields


def test_header_fields_module_level(img16_run: Path, tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(img16_run)
    stores = convert_run(run, tmp_path / "out")

    attrs = dict(zarr.open_group(str(stores[0]), mode="r").attrs)
    qf = set(attrs["quabo_fields"])
    # All four quabos x 5 fields = 20
    assert len(qf) == 20
    assert "quabo_0_pkt_num" in qf
    assert "quabo_3_tv_usec" in qf
    assert attrs["header_fields"] == []  # module-level has no single-level fields


# ── run_configs embedding ─────────────────────────────────────────────────────

def test_run_configs_embedded(tmp_path: Path) -> None:
    """Configs written alongside .pff files are embedded in zarr root attrs."""
    import json

    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run_dir = tmp_path / "conf_test.pffd"
    run_dir.mkdir()
    _make_ph256_pff(
        run_dir / "start_2024-01-01T00:00:00Z.dp_ph256.bpp_2.module_1.seqno_0.pff",
        n_frames=5,
    )
    # Use a filename that PanosetiRun stores as a raw dict (not a known Pydantic model)
    sw_cfg = {"version": "1.2.3", "build": "release"}
    (run_dir / "sw_info.json").write_text(json.dumps(sw_cfg))

    run = PanosetiRun(run_dir)
    stores = convert_run(run, tmp_path / "out", embed_configs=True)

    attrs = dict(zarr.open_group(str(stores[0]), mode="r").attrs)
    assert "run_configs" in attrs
    assert "sw_info" in attrs["run_configs"]
    assert attrs["run_configs"]["sw_info"]["version"] == "1.2.3"


# ── sidecar bundle ────────────────────────────────────────────────────────────

def test_sidecar_bundle_written(tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run_dir = tmp_path / "sc_test.pffd"
    run_dir.mkdir()
    _make_ph256_pff(
        run_dir / "start_2024-01-01T00:00:00Z.dp_ph256.bpp_2.module_1.seqno_0.pff",
        n_frames=5,
    )
    (run_dir / "obs_config.json").write_text('{"obs_name": "Test"}')
    (run_dir / "log.txt").write_text("run started\n")
    (run_dir / "collect_complete").write_text("")  # sentinel

    run = PanosetiRun(run_dir)
    convert_run(run, tmp_path / "out", write_sidecars=True)

    meta_dirs = list((tmp_path / "out").glob("*.panoseti-meta"))
    assert len(meta_dirs) == 1
    meta = meta_dirs[0]
    assert (meta / "manifest.json").exists()
    assert (meta / "configs" / "obs_config.json").exists()
    assert (meta / "logs" / "log.txt").exists()
    assert (meta / "sentinels" / "collect_complete").exists()


def test_sidecar_bundle_skipped(tmp_path: Path) -> None:
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run_dir = tmp_path / "nosc_test.pffd"
    run_dir.mkdir()
    _make_ph256_pff(
        run_dir / "start_2024-01-01T00:00:00Z.dp_ph256.bpp_2.module_1.seqno_0.pff",
        n_frames=5,
    )
    run = PanosetiRun(run_dir)
    convert_run(run, tmp_path / "out", write_sidecars=False)

    meta_dirs = list((tmp_path / "out").glob("*.panoseti-meta"))
    assert len(meta_dirs) == 0


# ── PanosetiZarrRun read-side wrapper ────────────────────────────────────────

def test_panoseti_zarr_run(ph256_run: Path, tmp_path: Path) -> None:
    import json

    from pypff.io2 import PanosetiRun
    from pypff.zarr import PanosetiZarrRun, convert_run

    # Use a raw-dict config (sw_info is not a known Pydantic model → stored as dict)
    (ph256_run / "sw_info.json").write_text(json.dumps({"obs_name": "RoundtripTest"}))

    run = PanosetiRun(ph256_run)
    out = tmp_path / "zrun_out"
    convert_run(run, out, write_sidecars=True, embed_configs=True)

    zrun = PanosetiZarrRun(out)
    products = zrun.list_products()
    assert len(products) == 1
    assert "dp_ph256" in products[0]

    store = zrun.get_product(products[0])
    assert store.data_product == "ph256"
    assert len(store) == 40
    assert store.timestamps().dtype == np.dtype("int64")
    assert store.header_fields != []
    assert store.quabo_fields == []

    # configs sourced from sidecar directory
    cfgs = zrun.configs
    assert "sw_info" in cfgs
    assert cfgs["sw_info"]["obs_name"] == "RoundtripTest"


# ── no ZarrUserWarning during conversion ─────────────────────────────────────

def test_no_zarr_user_warning(ph256_run: Path, tmp_path: Path) -> None:
    """convert_run must not emit ZarrUserWarning (consolidated metadata warning)."""
    import warnings

    from zarr.errors import MetadataValidationError  # noqa: F401 (ensure zarr imported)

    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert_run(run, tmp_path / "warn_out")

    zarr_warnings = [w for w in caught if issubclass(w.category, UserWarning)
                     and "consolidated" in str(w.message).lower()]
    assert zarr_warnings == [], f"Unexpected ZarrUserWarning(s): {zarr_warnings}"
