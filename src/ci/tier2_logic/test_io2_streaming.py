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
from pathlib import Path

import numpy as np
import pytest

from pypff.io2 import PFFSequence

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"
PH256_FILE = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
IMG16_FILE = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"


@pytest.fixture
def ph256_seq():
    if not PH256_FILE.exists():
        pytest.skip("PH256 test data not found.")
    return PFFSequence([PH256_FILE])


@pytest.fixture
def img16_seq():
    if not IMG16_FILE.exists():
        pytest.skip("IMG16 test data not found.")
    return PFFSequence([IMG16_FILE])


@pytest.fixture
def double_ph256():
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

def test_iter_yields_all_frames(ph256_seq):
    seq = ph256_seq
    frames = list(seq)
    assert len(frames) == len(seq)
    assert all(f.shape == seq.frame_config.image_shape for f in frames)


def test_iter_agrees_with_read_images_range(ph256_seq):
    seq = ph256_seq
    stacked = np.stack(list(seq))
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(stacked, bulk)


# ── iter_batches ──────────────────────────────────────────────

@pytest.mark.parametrize("batch_size", [1, 7, 50, 256])
def test_iter_batches_frame_count(ph256_seq, batch_size):
    seq = ph256_seq
    total = sum(b.shape[0] for b in seq.iter_batches(size=batch_size))
    assert total == len(seq)


@pytest.mark.parametrize("batch_size", [1, 50])
def test_iter_batches_data_equality(ph256_seq, batch_size):
    seq = ph256_seq
    batches = list(seq.iter_batches(size=batch_size))
    combined = np.concatenate(batches, axis=0)
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(combined, bulk)


def test_iter_batches_memory_bounded(ph256_seq):
    """Peak allocation from iter_batches should be << total data size."""
    seq = ph256_seq
    conf = seq.frame_config
    batch_size = 32
    per_batch_bytes = batch_size * conf.frame_size

    tracemalloc.start()
    total = 0
    for batch in seq.iter_batches(size=batch_size):
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


def test_iter_batches_with_timestamps(ph256_seq):
    seq = ph256_seq
    ts_from_batches = []
    for imgs, ts in seq.iter_batches(size=64, with_timestamps=True):
        assert ts.dtype == np.int64
        assert len(ts) == imgs.shape[0]
        ts_from_batches.append(ts)

    all_ts = np.concatenate(ts_from_batches)
    expected = seq.timestamps()
    np.testing.assert_array_equal(all_ts, expected)


def test_iter_batches_with_headers(ph256_seq):
    seq = ph256_seq
    for imgs, meta in seq.iter_batches(size=64, with_headers=True):
        assert isinstance(meta, dict)
        # should have at least some numeric fields from the header
        assert len(meta) > 0
        for v in meta.values():
            assert isinstance(v, np.ndarray)
            assert len(v) == imgs.shape[0]
        break  # just check first batch


def test_iter_batches_multi_file(double_ph256):
    seq = double_ph256
    total = sum(b.shape[0] for b in seq.iter_batches(size=50))
    assert total == len(seq)

    combined = np.concatenate(list(seq.iter_batches(size=50)), axis=0)
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(combined, bulk)


# ── iter_byte_range ───────────────────────────────────────────

def test_iter_byte_range_full_file(ph256_seq):
    seq = ph256_seq
    conf = seq.frame_config
    file_size = seq.file_paths[0].stat().st_size

    frames = list(seq.iter_byte_range(0, 0, file_size, batch_size=64))
    total = sum(f.shape[0] for f in frames)
    assert total == len(seq)


def test_iter_byte_range_half(ph256_seq):
    seq = ph256_seq
    conf = seq.frame_config
    file_size = seq.file_paths[0].stat().st_size
    # Split on a frame boundary so both halves are non-overlapping
    half_frames = len(seq) // 2
    half_bytes = half_frames * conf.frame_size

    from_start = sum(f.shape[0] for f in seq.iter_byte_range(0, 0, half_bytes, batch_size=64))
    from_end = sum(f.shape[0] for f in seq.iter_byte_range(0, half_bytes, file_size, batch_size=64))
    assert from_start + from_end == len(seq)


def test_iter_byte_range_data_correct(ph256_seq):
    seq = ph256_seq
    file_size = seq.file_paths[0].stat().st_size

    frames = np.concatenate(
        list(seq.iter_byte_range(0, 0, file_size, batch_size=128)), axis=0
    )
    bulk = seq.read_images_range(0)
    np.testing.assert_array_equal(frames, bulk)


def test_iter_byte_range_with_timestamps(ph256_seq):
    seq = ph256_seq
    file_size = seq.file_paths[0].stat().st_size
    ts_list = []
    for _, ts in seq.iter_byte_range(0, 0, file_size, batch_size=64, with_timestamps=True):
        ts_list.append(ts)
    all_ts = np.concatenate(ts_list)
    expected = seq.timestamps()
    np.testing.assert_array_equal(all_ts, expected)


# ── __getitem__ ───────────────────────────────────────────────

def test_getitem_int(ph256_seq):
    seq = ph256_seq
    frame = seq[0]
    assert frame.shape == seq.frame_config.image_shape
    assert frame.dtype == seq.frame_config.dtype


def test_getitem_int_negative(ph256_seq):
    seq = ph256_seq
    last = seq[-1]
    direct = seq[len(seq) - 1]
    np.testing.assert_array_equal(last, direct)


def test_getitem_slice(ph256_seq):
    seq = ph256_seq
    sliced = seq[0:10]
    assert sliced.shape == (10, *seq.frame_config.image_shape)
    bulk = seq.read_images_range(0, 10)
    np.testing.assert_array_equal(sliced, bulk)


def test_getitem_slice_stride(ph256_seq):
    seq = ph256_seq
    strided = seq[::2]
    expected = seq.read_images(np.arange(0, len(seq), 2))
    np.testing.assert_array_equal(strided, expected)


def test_getitem_out_of_bounds(ph256_seq):
    seq = ph256_seq
    with pytest.raises(IndexError):
        _ = seq[len(seq)]


# ── read_images (random access with sort) ────────────────────

def test_read_images_unsorted(ph256_seq):
    seq = ph256_seq
    indices = np.array([5, 0, 3, 1, 4, 2])
    out = seq.read_images(indices)
    for i, idx in enumerate(indices):
        np.testing.assert_array_equal(out[i], seq[idx])


def test_read_images_empty(ph256_seq):
    seq = ph256_seq
    out = seq.read_images(np.array([], dtype=np.int64))
    assert out.shape == (0, *seq.frame_config.image_shape)


# ── metadata: single-pass composite dtype ────────────────────

def test_get_metadata_arrays_sequential(ph256_seq):
    seq = ph256_seq
    meta = seq.get_metadata_arrays(["pkt_num", "tv_sec"])
    assert meta["pkt_num"].dtype == np.int64
    assert len(meta["pkt_num"]) == len(seq)
    # pkt_num should be monotonically increasing (mostly)
    assert meta["pkt_num"][0] >= 0


def test_get_metadata_arrays_indexed(ph256_seq):
    seq = ph256_seq
    indices = [0, 5, 2, 1]
    meta = seq.get_metadata_arrays(["pkt_num"], indices=indices)
    assert len(meta["pkt_num"]) == len(indices)
    # Results in original (caller) order
    for i, idx in enumerate(indices):
        h, _ = seq.get_frame(idx)
        assert int(meta["pkt_num"][i]) == h["pkt_num"]


def test_unix_t_ns_virtual_key(ph256_seq):
    seq = ph256_seq
    meta = seq.get_metadata_arrays(["unix_t_ns"])
    ts = meta["unix_t_ns"]
    assert ts.dtype == np.int64
    assert len(ts) == len(seq)
    # Should agree with timestamps()
    np.testing.assert_array_equal(ts, seq.timestamps())


def test_metadata_agrees_with_naive(ph256_seq):
    """Composite-dtype extraction must match naive frame-by-frame JSON parsing."""
    seq = ph256_seq
    assert seq.verify_metadata_offsets(num_frames=50)


# ── timestamps API ────────────────────────────────────────────

def test_timestamps_full(ph256_seq):
    seq = ph256_seq
    ts = seq.timestamps()
    assert ts.dtype == np.int64
    assert len(ts) == len(seq)


def test_timestamps_cached(ph256_seq):
    seq = ph256_seq
    ts1 = seq.timestamps()
    ts2 = seq.timestamps()
    assert ts1 is ts2  # same object returned


def test_timestamps_indexed(ph256_seq):
    seq = ph256_seq
    indices = np.array([0, 3, 1])
    ts_idx = seq.timestamps(indices=indices)
    ts_all = seq.timestamps()
    np.testing.assert_array_equal(ts_idx, ts_all[indices])


def test_timestamp_at(ph256_seq):
    seq = ph256_seq
    ts_all = seq.timestamps()
    for i in [0, 1, len(seq) - 1]:
        assert seq.timestamp_at(i) == int(ts_all[i])


def test_seek_time(ph256_seq):
    seq = ph256_seq
    t0 = seq.timestamp_at(0)
    idx = seq.seek_time(t0)
    assert idx == 0


# ── LRU and context manager ───────────────────────────────────

def test_lru_cap_respected(double_ph256):
    seq = double_ph256
    # Access both files
    seq.read_images_range(0, 2)
    seq.read_images_range(len(seq) - 2, 2)
    assert len(seq._lru) <= seq._lru.capacity


def test_context_manager_closes(ph256_seq):
    seq = ph256_seq
    with seq:
        _ = seq[0]
    assert len(seq._lru) == 0


def test_lru_custom_cap():
    if not PH256_FILE.exists():
        pytest.skip("PH256 test data not found.")
    seq = PFFSequence([PH256_FILE], max_open_files=1)
    assert seq._lru.capacity == 1
    _ = seq[0]
    assert len(seq._lru) <= 1


# ── module-level header tests ─────────────────────────────────

def test_module_get_frame_dict(img16_seq):
    h, img = img16_seq.get_frame(0)
    assert isinstance(h, dict)
    assert "quabo_0" in h
    assert "pkt_num" in h["quabo_0"]


def test_module_get_frame_validated(img16_seq):
    from pypff.models import ModuleHeader
    h, _ = img16_seq.get_frame_validated(0)
    assert isinstance(h, ModuleHeader)
    assert h.quabo_0.pkt_num >= 0


def test_module_unix_t_ns(img16_seq):
    ts = img16_seq.timestamps()
    assert ts.dtype == np.int64
    assert len(ts) == len(img16_seq)
    assert ts[0] > 0
