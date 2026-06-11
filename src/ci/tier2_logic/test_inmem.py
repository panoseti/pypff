"""
Tests for pypff.zarr.sequence_to_dataset — in-memory L0-layout Dataset builder.

All tests use synthetic PFF data generated in tmp_path to avoid any dependency
on external sample files.  The fixture helpers are intentionally identical to
those in test_zarr_roundtrip.py so the two test suites exercise the same data.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# sequence_to_dataset requires xarray; skip the whole module if not installed.
xr = pytest.importorskip("xarray", reason="xarray not installed; run: uv sync --extra zarr")


# ── fixture helpers ──────────────────────────────────────────────────────────


def _make_ph256_pff(path: Path, n_frames: int = 30, start_idx: int = 0) -> None:
    """Single-level (ph256) PFF: 16×16 int16 payload, single-quabo header."""
    header_base = (
        '{"quabo_num":          0, "pkt_num": %10d, "pkt_tai":          0, '
        '"pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec":          0}\n\n*'
    )
    payload = np.arange(256, dtype=np.int16).reshape(16, 16).tobytes()
    with open(path, "wb") as f:
        for i in range(start_idx, start_idx + n_frames):
            h = header_base % (i, i * 1_000_000)
            f.write(h.encode() + payload)


def _make_img16_pff(path: Path, n_frames: int = 24, start_idx: int = 0) -> None:
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
                i, 0, i * 1_000_000,
                i, 0, i * 1_000_000,
                i, 0, i * 1_000_000,
                i, 0, i * 1_000_000,
            )
            f.write(h.encode() + payload)


@pytest.fixture
def ph256_seq(tmp_path: Path):
    """PFFSequence over two 30-frame ph256 files (60 frames total)."""
    from pypff.io2 import PFFSequence

    run_dir = tmp_path / "ph256_run.pffd"
    run_dir.mkdir()
    p0 = run_dir / "start_2024-01-01T00:00:00Z.dp_ph256.bpp_2.module_1.seqno_0.pff"
    p1 = run_dir / "start_2024-01-01T00:00:10Z.dp_ph256.bpp_2.module_1.seqno_1.pff"
    _make_ph256_pff(p0, n_frames=30, start_idx=0)
    _make_ph256_pff(p1, n_frames=30, start_idx=30)
    return PFFSequence([p0, p1])


@pytest.fixture
def img16_seq(tmp_path: Path):
    """PFFSequence over a single 24-frame img16 file."""
    from pypff.io2 import PFFSequence

    run_dir = tmp_path / "img16_run.pffd"
    run_dir.mkdir()
    p = run_dir / "start_2024-01-01T00:00:00Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    _make_img16_pff(p, n_frames=24, start_idx=0)
    return PFFSequence([p])


# ── 1. Basic shape ────────────────────────────────────────────────────────────


def test_basic_shape_ph256(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    T = len(ph256_seq)
    assert ds["images"].shape == (T, 16, 16)
    assert ds["unix_t_ns"].shape == (T,)
    for key in ph256_seq.metadata_offsets:
        zarr_name = key.replace(".", "_")
        assert zarr_name in ds, f"Header array '{zarr_name}' missing from dataset"
        assert ds[zarr_name].shape == (T,)


def test_basic_shape_img16(img16_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(img16_seq)
    T = len(img16_seq)
    assert ds["images"].shape == (T, 32, 32)
    assert ds["unix_t_ns"].shape == (T,)


# ── 2. Root attrs present ─────────────────────────────────────────────────────


def test_root_attrs_present(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    assert ds.attrs["panoseti_pff_zarr_version"] == "1.1"
    assert ds.attrs["data_product"] == "ph256"
    assert "module" in ds.attrs
    assert "frame_config" in ds.attrs
    assert ds.attrs["total_frames"] == len(ph256_seq)
    assert "header_fields" in ds.attrs
    assert "quabo_fields" in ds.attrs
    assert "source_pff_files" in ds.attrs
    assert isinstance(ds.attrs["source_pff_files"], list)


def test_root_attrs_frame_config_keys(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    fc = ds.attrs["frame_config"]
    for key in ("header_size", "payload_size", "frame_size", "image_shape",
                "dtype_str", "bytes_per_pixel", "format_name"):
        assert key in fc, f"frame_config missing key '{key}'"


# ── 3. Subrange by frame index ────────────────────────────────────────────────


def test_subrange_frame_index(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq, start=10, stop=20)
    assert len(ds.time) == 10
    assert ds["images"].shape[0] == 10
    assert ds["unix_t_ns"].shape == (10,)


def test_subrange_partial_start(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq, start=55)
    # 60 total frames, start=55 → 5 frames
    assert len(ds.time) == 5


def test_subrange_empty(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq, start=10, stop=10)
    assert len(ds.time) == 0


# ── 4. Step / decimation ──────────────────────────────────────────────────────


def test_step_decimation(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    T = len(ph256_seq)
    step = 3
    ds = sequence_to_dataset(ph256_seq, step=step)
    expected = math.ceil(T / step)
    assert len(ds.time) == expected


def test_step_with_range(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq, start=0, stop=30, step=5)
    assert len(ds.time) == 6  # 0,5,10,15,20,25


# ── 5. Time-range selection ───────────────────────────────────────────────────


def test_time_range_selection(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    all_ts = ph256_seq.timestamps()
    t0 = int(all_ts[10])
    t1 = int(all_ts[20])

    ds = sequence_to_dataset(ph256_seq, start_ns=t0, stop_ns=t1)
    ts_selected = ds["unix_t_ns"].values
    assert len(ts_selected) > 0
    assert np.all(ts_selected >= t0)
    assert np.all(ts_selected < t1)


def test_time_range_start_ns_only(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    all_ts = ph256_seq.timestamps()
    t0 = int(all_ts[20])

    ds = sequence_to_dataset(ph256_seq, start_ns=t0)
    ts_selected = ds["unix_t_ns"].values
    assert len(ts_selected) > 0
    assert np.all(ts_selected >= t0)


# ── 6. Dim names ──────────────────────────────────────────────────────────────


def test_dim_names_images(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    assert ds["images"].dims == ("time", "y", "x")


def test_dim_names_unix_t_ns(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    assert ds["unix_t_ns"].dims == ("time",)


def test_dim_names_header_fields(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    for key in ph256_seq.metadata_offsets:
        zarr_name = key.replace(".", "_")
        assert ds[zarr_name].dims == ("time",), (
            f"Header array '{zarr_name}' has wrong dims: {ds[zarr_name].dims}"
        )


# ── 7. Array dtypes ───────────────────────────────────────────────────────────


def test_header_dtypes_single_level(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    assert ds["pkt_num"].dtype == np.dtype("uint32")
    assert ds["pkt_tai"].dtype == np.dtype("uint16")
    assert ds["pkt_nsec"].dtype == np.dtype("uint32")
    assert ds["tv_sec"].dtype == np.dtype("int64")
    assert ds["tv_usec"].dtype == np.dtype("uint32")
    assert ds["quabo_num"].dtype == np.dtype("uint8")


def test_header_dtypes_module_level(img16_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(img16_seq)
    assert ds["quabo_0_pkt_num"].dtype == np.dtype("uint32")
    assert ds["quabo_0_pkt_tai"].dtype == np.dtype("uint16")
    assert ds["quabo_0_pkt_nsec"].dtype == np.dtype("uint32")
    assert ds["quabo_0_tv_sec"].dtype == np.dtype("int64")
    assert ds["quabo_0_tv_usec"].dtype == np.dtype("uint32")


def test_unix_t_ns_dtype(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    assert ds["unix_t_ns"].dtype == np.dtype("int64")


# ── 8. Images match PFF data ──────────────────────────────────────────────────


def test_images_match_pff(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    expected = ph256_seq.read_images_range(0)
    np.testing.assert_array_equal(ds["images"].values, expected)


def test_images_subrange_match_pff(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq, start=5, stop=15)
    expected = ph256_seq.read_images_range(5, 10)
    np.testing.assert_array_equal(ds["images"].values, expected)


# ── 9. Timestamps match PFF data ──────────────────────────────────────────────


def test_timestamps_match_pff(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    expected_ts = ph256_seq.timestamps()
    np.testing.assert_array_equal(ds["unix_t_ns"].values, expected_ts)


# ── 10. total_frames attr reflects full sequence, not slice ──────────────────


def test_total_frames_attr_is_full_sequence(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    T_full = len(ph256_seq)
    ds = sequence_to_dataset(ph256_seq, start=10, stop=20)
    assert ds.attrs["total_frames"] == T_full
    assert len(ds.time) == 10  # slice has 10 frames


# ── 11. run_configs embedded when provided ────────────────────────────────────


def test_run_configs_embedded(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    cfgs = {"obs_config": {"obs_name": "TestObs"}}
    ds = sequence_to_dataset(ph256_seq, run_configs=cfgs)
    assert "run_configs" in ds.attrs
    assert ds.attrs["run_configs"]["obs_config"]["obs_name"] == "TestObs"


def test_run_configs_absent_when_none(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    assert "run_configs" not in ds.attrs


# ── 12. Module-level quabo fields discoverability ────────────────────────────


def test_quabo_fields_attr_module_level(img16_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(img16_seq)
    qf = set(ds.attrs["quabo_fields"])
    # 4 quabos × 5 fields = 20
    assert len(qf) == 20
    assert "quabo_0_pkt_num" in qf
    assert "quabo_3_tv_usec" in qf
    assert ds.attrs["header_fields"] == []


def test_header_fields_attr_single_level(ph256_seq) -> None:
    from pypff.zarr import sequence_to_dataset

    ds = sequence_to_dataset(ph256_seq)
    hf = set(ds.attrs["header_fields"])
    assert hf == {"pkt_num", "pkt_tai", "pkt_nsec", "tv_sec", "tv_usec", "quabo_num"}
    assert ds.attrs["quabo_fields"] == []
