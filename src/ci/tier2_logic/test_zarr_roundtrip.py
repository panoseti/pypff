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

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", reason="zarr not installed; run: uv sync --extra zarr")


# ── fixture helpers ──────────────────────────────────────────────────────────

def _make_ph256_pff(path: Path, n_frames: int = 20, start_idx: int = 0) -> None:
    """Single-level (ph256) PFF: 16×16 int16 payload, quabo_num header."""
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
    """Module-level (img16) PFF: 32×32 uint16 payload, quabo_0..3 header."""
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
def ph256_run(tmp_path):
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
def img16_run(tmp_path):
    run_dir = tmp_path / "img16_test.pffd"
    run_dir.mkdir()
    _make_img16_pff(
        run_dir / "start_2024-01-01T00:00:00Z.dp_img16.bpp_2.module_1.seqno_0.pff",
        n_frames=15, start_idx=0,
    )
    return run_dir


# ── roundtrip: images ─────────────────────────────────────────────────────────

def test_images_roundtrip_ph256(ph256_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")
    assert len(stores) == 1

    pff_seq = run.get_product(run.list_products()[0])
    expected = pff_seq.read_images_range(0)

    store = zarr.open_group(str(stores[0]), mode="r")
    np.testing.assert_array_equal(store["images"][:], expected)


def test_images_roundtrip_img16(img16_run, tmp_path):
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

def test_timestamps_match_pff(ph256_run, tmp_path):
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

def test_header_values_match_pff(ph256_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    pff_seq = run.get_product(run.list_products()[0])
    pff_meta = pff_seq.get_metadata_arrays(["pkt_num"])

    store = zarr.open_group(str(stores[0]), mode="r")
    zarr_pkt_num = store["headers"]["pkt_num"][:].astype(np.int64)
    np.testing.assert_array_equal(zarr_pkt_num, pff_meta["pkt_num"])


# ── header dtypes ─────────────────────────────────────────────────────────────

def test_header_dtypes_single_level(ph256_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    headers = store["headers"]

    expected = {
        "quabo_num": np.dtype("uint8"),
        "pkt_num":   np.dtype("uint32"),
        "pkt_tai":   np.dtype("uint16"),
        "pkt_nsec":  np.dtype("uint32"),
        "tv_sec":    np.dtype("int64"),
        "tv_usec":   np.dtype("uint32"),
    }
    for field, expected_dtype in expected.items():
        if field in headers:
            assert headers[field].dtype == expected_dtype, (
                f"headers/{field}: got {headers[field].dtype}, expected {expected_dtype}"
            )


def test_header_dtypes_module_level(img16_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(img16_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    q0 = store["headers"]["quabo_0"]

    assert q0["pkt_num"].dtype == np.dtype("uint32")
    assert q0["pkt_tai"].dtype == np.dtype("uint16")
    assert q0["pkt_nsec"].dtype == np.dtype("uint32")
    assert q0["tv_sec"].dtype == np.dtype("int64")
    assert q0["tv_usec"].dtype == np.dtype("uint32")


# ── module-level header structure ─────────────────────────────────────────────

def test_module_header_subgroups(img16_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(img16_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    headers = store["headers"]

    # All four quabo sub-groups present
    for qi in range(4):
        assert f"quabo_{qi}" in headers, f"headers/quabo_{qi} missing"
        assert "pkt_num" in headers[f"quabo_{qi}"]

    # No quabo_num at top level (that's single-level only)
    assert "quabo_num" not in headers


# ── ZarrWriter protocol ───────────────────────────────────────────────────────

def test_no_op_writer_satisfies_protocol():
    from pypff.zarr import ZarrWriter

    class RecordingWriter:
        def __init__(self):
            self.calls: list[str] = []

        def create_store(self, path):
            self.calls.append("create_store")
            return {}

        def create_array(self, root_group, name, shape, chunks, dtype):
            self.calls.append(f"create_array:{name}")
            return {}

        def write_slice(self, array, slices, data):
            self.calls.append("write_slice")

        def set_attrs(self, obj, attrs):
            self.calls.append("set_attrs")

        def finalize(self, path):
            self.calls.append("finalize")

    assert isinstance(RecordingWriter(), ZarrWriter)


# ── store attributes ──────────────────────────────────────────────────────────

def test_root_attrs_present(ph256_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    attrs = dict(store.attrs)
    assert "panoseti_pff_zarr_version" in attrs
    assert "frame_config" in attrs
    assert attrs["data_product"] == "ph256"
    assert attrs["total_frames"] == 40  # 2 files × 20 frames


def test_unix_t_ns_attrs(ph256_run, tmp_path):
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    store = zarr.open_group(str(stores[0]), mode="r")
    ts_attrs = dict(store["unix_t_ns"].attrs)
    assert "ns_since_epoch" in ts_attrs
    assert "_ARRAY_DIMENSIONS" in ts_attrs


# ── xarray integration ────────────────────────────────────────────────────────

def test_xarray_open(ph256_run, tmp_path):
    pytest.importorskip("xarray")
    from pypff.io2 import PanosetiRun
    from pypff.zarr import convert_run

    run = PanosetiRun(ph256_run)
    stores = convert_run(run, tmp_path / "out")

    import xarray as xr
    ds = xr.open_zarr(str(stores[0]))
    assert "images" in ds
    assert "unix_t_ns" in ds
    assert ds.images.dims == ("time", "y", "x")
    assert ds["unix_t_ns"].dtype == np.dtype("int64")
