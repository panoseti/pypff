"""
Streaming API tests for PFFSequence.

Verifies:
- iter_batches yields correct frames in order with bounded memory
- __iter__ works for per-frame streaming
- iter_byte_range yields the correct subrange
- read_images / read_images_range produce identical results to iteration
- unix_t_ns virtual key round-trips correctly
- timestamps() and timestamp_at() agree
- LRU mmap cap is respected
- Context manager closes handles
"""
import shutil
import tempfile
import tracemalloc
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from pypff.io2 import PFFSequence

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"
PH256_FILE = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
IMG16_FILE = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"


@pytest.fixture
def ph256_seq() -> PFFSequence:
    if not PH256_FILE.exists():
        pytest.skip("PH256 test data not found.")
    return PFFSequence([PH256_FILE])


@pytest.fixture
def img16_seq() -> PFFSequence:
    if not IMG16_FILE.exists():
        pytest.skip("IMG16 test data not found.")
    return PFFSequence([IMG16_FILE])


@pytest.fixture
def double_ph256() -> Generator[PFFSequence]:
    """Two-file sequence made by copying the ph256 fixture file."""
    if not PH256_FILE.exists():
        pytest.skip("PH256 test data not found.")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        f0 = tmp / "test.dp_ph256.module_1.seqno_0.pff"
        f1 = tmp / "test.dp_ph256.module_1.seqno_1.pff"
        shutil.copy(PH256_FILE, f0)
        shutil.copy(PH256_FILE, f1)
        yield PFFSequence([f0, f1])


# ── basic iteration ──────────────────────────────────────────

def test_iter_yields_all_frames(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    assert seq.frame_config is not None
    frames = list(seq)
    assert len(frames) == len(seq)
    assert all(f.shape == seq.frame_config.image_shape for f in frames)


def test_iter_agrees_with_read_images_range(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    stacked = np.stack(list(seq))
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(stacked, bulk)


# ── iter_batches ──────────────────────────────────────────────

@pytest.mark.parametrize("batch_size", [1, 7, 50, 256])
def test_iter_batches_frame_count(ph256_seq: PFFSequence, batch_size: int) -> None:
    seq = ph256_seq
    total = 0
    for b in seq.iter_batches(size=batch_size):
        assert isinstance(b, np.ndarray)
        total += b.shape[0]
    assert total == len(seq)


@pytest.mark.parametrize("batch_size", [1, 50])
def test_iter_batches_data_equality(ph256_seq: PFFSequence, batch_size: int) -> None:
    seq = ph256_seq
    batches = []
    for b in seq.iter_batches(size=batch_size):
        assert isinstance(b, np.ndarray)
        batches.append(b)
    combined = np.concatenate(batches, axis=0)
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(combined, bulk)


def test_iter_batches_memory_bounded(ph256_seq: PFFSequence) -> None:
    """Peak allocation from iter_batches should be << total data size."""
    seq = ph256_seq
    conf = seq.frame_config
    batch_size = 32
    assert conf is not None
    per_batch_bytes = batch_size * conf.frame_size

    tracemalloc.start()
    total = 0
    for batch in seq.iter_batches(size=batch_size):
        assert isinstance(batch, np.ndarray)
        total += batch.shape[0]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert total == len(seq)
    # Peak should be well under the size of the full sequence
    full_size = len(seq) * conf.frame_size
    # Allow generous headroom; key assertion is that we're NOT loading everything
    if full_size > per_batch_bytes * 4:
        assert peak < full_size // 2, (
            f"Peak allocation {peak / 1024:.1f} KB is suspiciously close to "
            f"full sequence size {full_size / 1024:.1f} KB"
        )


def test_iter_batches_with_timestamps(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    ts_from_batches = []
    for imgs, ts in seq.iter_batches(size=64, with_timestamps=True):
        assert ts.dtype == np.int64
        assert len(ts) == imgs.shape[0]
        ts_from_batches.append(ts)

    all_ts = np.concatenate(ts_from_batches)
    expected = seq.timestamps()
    np.testing.assert_array_equal(all_ts, expected)


def test_iter_batches_with_headers(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    for imgs, meta in seq.iter_batches(size=64, with_headers=True):
        assert isinstance(meta, dict)
        # should have at least some numeric fields from the header
        assert len(meta) > 0
        for v in meta.values():
            assert isinstance(v, np.ndarray)
            assert len(v) == imgs.shape[0]
        break  # just check first batch


def test_iter_batches_multi_file(double_ph256: PFFSequence) -> None:
    seq = double_ph256
    total = 0
    for b in seq.iter_batches(size=50):
        assert isinstance(b, np.ndarray)
        total += b.shape[0]
    assert total == len(seq)

    batches = []
    for b in seq.iter_batches(size=50):
        assert isinstance(b, np.ndarray)
        batches.append(b)
    combined = np.concatenate(batches, axis=0)
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(combined, bulk)


# ── iter_byte_range ───────────────────────────────────────────

def test_iter_byte_range_full_file(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    file_size = seq.file_paths[0].stat().st_size

    total = 0
    for f in seq.iter_byte_range(0, 0, file_size, batch_size=64):
        assert isinstance(f, np.ndarray)
        total += f.shape[0]
    assert total == len(seq)


def test_iter_byte_range_half(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    conf = seq.frame_config
    assert conf is not None
    file_size = seq.file_paths[0].stat().st_size
    # Split on a frame boundary so both halves are non-overlapping
    half_frames = len(seq) // 2
    half_bytes = half_frames * conf.frame_size

    from_start = 0
    for f in seq.iter_byte_range(0, 0, half_bytes, batch_size=64):
        assert isinstance(f, np.ndarray)
        from_start += f.shape[0]

    from_end = 0
    for f in seq.iter_byte_range(0, half_bytes, file_size, batch_size=64):
        assert isinstance(f, np.ndarray)
        from_end += f.shape[0]

    assert from_start + from_end == len(seq)


def test_iter_byte_range_data_correct(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    file_size = seq.file_paths[0].stat().st_size

    batches = []
    for f in seq.iter_byte_range(0, 0, file_size, batch_size=128):
        assert isinstance(f, np.ndarray)
        batches.append(f)
    frames = np.concatenate(batches, axis=0)
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(frames, bulk)


def test_iter_byte_range_with_timestamps(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    file_size = seq.file_paths[0].stat().st_size
    ts_list = []
    for _, ts in seq.iter_byte_range(0, 0, file_size, batch_size=64, with_timestamps=True):
        ts_list.append(ts)
    all_ts = np.concatenate(ts_list)
    expected = seq.timestamps()
    np.testing.assert_array_equal(all_ts, expected)


# ── __getitem__ ───────────────────────────────────────────────

def test_getitem_int(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    frame = seq[0]
    assert seq.frame_config is not None
    assert frame.shape == seq.frame_config.image_shape
    assert frame.dtype == seq.frame_config.dtype


def test_getitem_int_negative(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    last = seq[-1]
    direct = seq[len(seq) - 1]
    np.testing.assert_array_equal(last, direct)


def test_getitem_slice(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    sliced = seq[0:10]
    assert seq.frame_config is not None
    assert sliced.shape == (10, *seq.frame_config.image_shape)
    bulk = seq.read_images_range(0, 10)
    np.testing.assert_array_equal(sliced, bulk)


def test_getitem_slice_stride(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    strided = seq[::2]
    expected = seq.read_images(np.arange(0, len(seq), 2))
    np.testing.assert_array_equal(strided, expected)


def test_getitem_out_of_bounds(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    with pytest.raises(IndexError):
        _ = seq[len(seq)]


# ── read_images (random access with sort) ────────────────────

def test_read_images_unsorted(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    indices = np.array([5, 0, 3, 1, 4, 2])
    out = seq.read_images(indices)
    for i, idx in enumerate(indices):
        np.testing.assert_array_equal(out[i], seq[idx])


def test_read_images_empty(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    out = seq.read_images(np.array([], dtype=np.int64))
    assert seq.frame_config is not None
    assert out.shape == (0, *seq.frame_config.image_shape)


# ── metadata: single-pass composite dtype ────────────────────

def test_get_metadata_arrays_sequential(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    meta = seq.get_metadata_arrays(["pkt_num", "tv_sec"])
    assert meta["pkt_num"].dtype == np.int64
    assert len(meta["pkt_num"]) == len(seq)
    # pkt_num should be monotonically increasing (mostly)
    assert meta["pkt_num"][0] >= 0


def test_get_metadata_arrays_indexed(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    indices = [0, 5, 2, 1]
    meta = seq.get_metadata_arrays(["pkt_num"], indices=indices)
    assert len(meta["pkt_num"]) == len(indices)
    # Results in original (caller) order
    for i, idx in enumerate(indices):
        h, _ = seq.get_frame(idx)
        assert int(meta["pkt_num"][i]) == h["pkt_num"]


def test_unix_t_ns_virtual_key(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    meta = seq.get_metadata_arrays(["unix_t_ns"])
    ts = meta["unix_t_ns"]
    assert ts.dtype == np.int64
    assert len(ts) == len(seq)
    # Should agree with timestamps()
    np.testing.assert_array_equal(ts, seq.timestamps())


def test_metadata_agrees_with_naive(ph256_seq: PFFSequence) -> None:
    """Composite-dtype extraction must match naive frame-by-frame JSON parsing."""
    seq = ph256_seq
    assert seq.verify_metadata_offsets(num_frames=50)


# ── timestamps API ────────────────────────────────────────────

def test_timestamps_full(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    ts = seq.timestamps()
    assert ts.dtype == np.int64
    assert len(ts) == len(seq)


def test_timestamps_cached(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    ts1 = seq.timestamps()
    ts2 = seq.timestamps()
    assert ts1 is ts2  # same object returned


def test_timestamps_indexed(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    indices = np.array([0, 3, 1])
    ts_idx = seq.timestamps(indices=indices)
    ts_all = seq.timestamps()
    np.testing.assert_array_equal(ts_idx, ts_all[indices])


def test_timestamp_at(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    ts_all = seq.timestamps()
    for i in [0, 1, len(seq) - 1]:
        assert seq.timestamp_at(i) == int(ts_all[i])


def test_seek_time(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    t0 = seq.timestamp_at(0)
    idx = seq.seek_time(t0)
    assert idx == 0


# ── timestamps_at (batch vectorized) ─────────────────────────

def test_timestamps_at_matches_timestamp_at(ph256_seq: PFFSequence) -> None:
    """timestamps_at must return the same values as repeated timestamp_at calls."""
    seq = ph256_seq
    indices = np.array([0, 1, len(seq) - 1])
    batch = seq.timestamps_at(indices)
    assert batch.dtype == np.int64
    assert len(batch) == len(indices)
    for i, idx in enumerate(indices):
        assert batch[i] == seq.timestamp_at(int(idx))


def test_timestamps_at_matches_timestamps_full(ph256_seq: PFFSequence) -> None:
    """timestamps_at with all indices must match the full timestamps() cache."""
    seq = ph256_seq
    all_indices = np.arange(len(seq), dtype=np.int64)
    batch = seq.timestamps_at(all_indices)
    expected = seq.timestamps()
    np.testing.assert_array_equal(batch, expected)


def test_timestamps_at_unsorted_indices(ph256_seq: PFFSequence) -> None:
    """Results must come back in caller's original order, not sorted order."""
    seq = ph256_seq
    indices = np.array([len(seq) - 1, 0, 2, 1], dtype=np.int64)
    batch = seq.timestamps_at(indices)
    ts_all = seq.timestamps()
    np.testing.assert_array_equal(batch, ts_all[indices])


def test_timestamps_at_single_index(ph256_seq: PFFSequence) -> None:
    """Edge case: single-element batch should equal timestamp_at."""
    seq = ph256_seq
    result = seq.timestamps_at(np.array([3], dtype=np.int64))
    assert len(result) == 1
    assert result[0] == seq.timestamp_at(3)


def test_timestamps_at_strided_sample(ph256_seq: PFFSequence) -> None:
    """Simulate the extract_timeline use-case: strided sample across the full sequence."""
    seq = ph256_seq
    stride = max(len(seq) // 10, 1)
    indices = np.arange(0, len(seq), stride, dtype=np.int64)
    batch = seq.timestamps_at(indices)
    ts_all = seq.timestamps()
    np.testing.assert_array_equal(batch, ts_all[indices])


# ── indexed get_metadata_arrays correctness ───────────────────

def test_get_metadata_indexed_agrees_with_sequential(ph256_seq: PFFSequence) -> None:
    """
    The indexed path of get_metadata_arrays (now using _composite_extract + fancy indexing)
    must produce bit-identical results to the sequential bulk path for the same frames.

    Regression test for the bug where the indexed path used a Python byte-read loop
    instead of _composite_extract.
    """
    seq = ph256_seq
    keys = ["pkt_num", "tv_sec"]
    # Get all values via the fast sequential bulk path
    seq_result = seq.get_metadata_arrays(keys)
    # Get a subset via the indexed path
    indices = np.array([0, 1, 2, len(seq) // 2, len(seq) - 1], dtype=np.int64)
    idx_result = seq.get_metadata_arrays(keys, indices=indices)
    for k in keys:
        np.testing.assert_array_equal(
            idx_result[k], seq_result[k][indices],
            err_msg=f"Indexed path mismatch for key '{k}'"
        )


def test_get_metadata_indexed_unix_t_ns(ph256_seq: PFFSequence) -> None:
    """unix_t_ns virtual key must work correctly through the indexed path."""
    seq = ph256_seq
    indices = np.array([0, len(seq) // 2, len(seq) - 1], dtype=np.int64)
    idx_ts = seq.get_metadata_arrays(["unix_t_ns"], indices=indices)["unix_t_ns"]
    full_ts = seq.timestamps()
    np.testing.assert_array_equal(idx_ts, full_ts[indices])


# ── LRU and context manager ───────────────────────────────────

def test_lru_cap_respected(double_ph256: PFFSequence) -> None:
    seq = double_ph256
    # Access both files
    seq.read_images_range(0, 2)
    seq.read_images_range(len(seq) - 2, 2)
    assert len(seq._lru) <= seq._lru.capacity


def test_context_manager_closes(ph256_seq: PFFSequence) -> None:
    seq = ph256_seq
    with seq:
        _ = seq[0]
    assert len(seq._lru) == 0


def test_lru_custom_cap() -> None:
    if not PH256_FILE.exists():
        pytest.skip("PH256 test data not found.")
    seq = PFFSequence([PH256_FILE], max_open_files=1)
    assert seq._lru.capacity == 1
    _ = seq[0]
    assert len(seq._lru) <= 1


# ── module-level header tests ─────────────────────────────────

def test_module_get_frame_dict(img16_seq: PFFSequence) -> None:
    h, _img = img16_seq.get_frame(0)
    assert isinstance(h, dict)
    assert "quabo_0" in h
    assert "pkt_num" in h["quabo_0"]


def test_module_get_frame_validated(img16_seq: PFFSequence) -> None:
    from pypff.models import ModuleHeader
    h, _ = img16_seq.get_frame_validated(0)
    assert isinstance(h, ModuleHeader)
    assert h.quabo_0.pkt_num >= 0


def test_module_unix_t_ns(img16_seq: PFFSequence) -> None:
    ts = img16_seq.timestamps()
    assert ts.dtype == np.int64
    assert len(ts) == len(img16_seq)
    assert ts[0] > 0
