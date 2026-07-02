import concurrent.futures
from pathlib import Path

import numpy as np
import pytest

from pypff.io2 import PFFSequence


def create_dummy_pff(path: Path, n_frames: int = 10, is_module: bool = False, start_idx: int = 0) -> None:
    if is_module:
        # Fixed-width format strings (mimicking PanoSETI's space-padded JSON)
        header_base = (
            '{"quabo_0": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}, '
            '"quabo_1": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}, '
            '"quabo_2": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}, '
            '"quabo_3": {"pkt_num": %10d, "pkt_tai": %10d, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}}\n\n*'
        )
    else:
        header_base = '{"quabo_num": 0, "pkt_num": %10d, "pkt_tai": 0, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}\n\n*'
    
    payload = np.arange(1024, dtype=np.uint16).reshape(32, 32).tobytes()
    
    with open(path, 'wb') as f:
        for i in range(start_idx, start_idx + n_frames):
            if is_module:
                h = header_base % (i, 0, i*1000, i, 0, i*1000, i, 0, i*1000, i, 0, i*1000)
            else:
                h = header_base % (i, i*1000)
            f.write(h.encode() + payload)

@pytest.fixture
def dummy_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "test_run.pffd"
    run_dir.mkdir()
    create_dummy_pff(run_dir / "start_2024-01-01T00:00:00Z.dp_img16.bpp_2.module_1.seqno_0.pff", n_frames=10, start_idx=0)
    create_dummy_pff(run_dir / "start_2024-01-01T00:00:10Z.dp_img16.bpp_2.module_1.seqno_1.pff", n_frames=10, start_idx=10)
    (run_dir / "obs_config.json").write_text('{"name": "test_obs", "domes": []}')
    (run_dir / "data_config.json").write_text('{"run_type": "test", "image": {"integration_time_usec": 1000, "pe_threshold": 1.0, "quabo_sample_size": 16}}')
    return run_dir

def test_pffsequence_slicing(dummy_run: Path) -> None:
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    assert len(seq) == 20
    img0 = seq[0]
    assert img0.shape == (32, 32)
    chunk = seq[0:5]
    assert chunk.shape == (5, 32, 32)
    strided = seq[0:20:2]
    assert strided.shape == (10, 32, 32)
    last = seq[-1]
    assert np.array_equal(last, seq[19])
    rev = seq[5:0:-1]
    assert rev.shape == (5, 32, 32)

def read_frame_sum(s: PFFSequence, i: int) -> int:
    _, img = s.get_frame(i)
    return int(img.sum())

def test_pffsequence_multiprocessing(dummy_run: Path) -> None:
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(read_frame_sum, seq, i) for i in range(len(seq))]
        results = [f.result() for f in futures]
    assert len(results) == 20

def test_pffsequence_timing_and_seek(dummy_run: Path) -> None:
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    t0 = seq.timestamp_at(0)
    t1 = seq.timestamp_at(1)
    assert t1 > t0
    idx = seq.seek_time(t0 + 500)
    assert idx in [0, 1]


def test_timestamps_at_synthetic(dummy_run: Path) -> None:
    """timestamps_at on synthetic data (quabo format) agrees with timestamp_at."""
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    indices = np.arange(len(seq), dtype=np.int64)
    batch = seq.timestamps_at(indices)
    assert batch.dtype == np.int64
    assert len(batch) == len(seq)
    # Every element must match the individual timestamp_at result
    for i in range(len(seq)):
        assert batch[i] == seq.timestamp_at(i), f"Mismatch at index {i}"


def test_timestamps_at_unsorted_order(dummy_run: Path) -> None:
    """timestamps_at must preserve caller order, not sort order."""
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    indices = np.array([len(seq) - 1, 0, 5, 3], dtype=np.int64)
    batch = seq.timestamps_at(indices)
    for i, idx in enumerate(indices):
        assert batch[i] == seq.timestamp_at(int(idx)), \
            f"Out-of-order result wrong at position {i} (global frame {idx})"


def test_timestamps_at_multi_file_span(dummy_run: Path) -> None:
    """timestamps_at must work correctly when indices span multiple files."""
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    # Pick indices from both files (file 0 has frames 0..9, file 1 has 10..19)
    indices = np.array([0, 9, 10, 19], dtype=np.int64)
    batch = seq.timestamps_at(indices)
    for i, idx in enumerate(indices):
        assert batch[i] == seq.timestamp_at(int(idx)), \
            f"Cross-file mismatch at index {idx}"


def test_pffsequence_stress_many_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "stress_run.pffd"
    run_dir.mkdir()
    n_files, frames_per_file = 50, 2
    for i in range(n_files):
        path = run_dir / f"start_2024.dp_img16.bpp_2.module_1.seqno_{i}.pff"
        create_dummy_pff(path, n_frames=frames_per_file, start_idx=i*frames_per_file)
    files = sorted(list(run_dir.glob("*.pff")), key=lambda x: int(x.name.split('seqno_')[1].split('.')[0]))
    seq = PFFSequence(files)
    indices = np.random.choice(len(seq), 20, replace=False)
    imgs = seq.read_images(indices)
    assert imgs.shape == (20, 32, 32)


from unittest.mock import patch

def test_timestamps_at_sparse_vs_dense(tmp_path: Path) -> None:
    """Verify that sparse access avoids _composite_extract and dense access uses it."""
    run_dir = tmp_path / "scale_run.pffd"
    run_dir.mkdir()
    path = run_dir / "start_2024.dp_img16.bpp_2.module_1.seqno_0.pff"
    
    # 2000 frames is enough to test the threshold (SPARSE_THRESHOLD = 100)
    create_dummy_pff(path, n_frames=2000)
    seq = PFFSequence([path])
    
    with patch.object(seq, '_composite_extract', wraps=seq._composite_extract) as mock_extract:
        # DENSE access: span = 50, count = 50. (span <= 100 * count) -> 50 <= 5000 -> True
        indices_dense = np.arange(0, 50, dtype=np.int64)
        ts_dense = seq.timestamps_at(indices_dense)
        assert len(ts_dense) == 50
        assert mock_extract.call_count == 1
        
        mock_extract.reset_mock()
        
        # SPARSE access: span = 1999, count = 4. (span <= 100 * count) -> 1999 <= 400 -> False
        indices_sparse = np.array([0, 500, 1000, 1999], dtype=np.int64)
        ts_sparse = seq.timestamps_at(indices_sparse)
        assert len(ts_sparse) == 4
        assert mock_extract.call_count == 0  # Should use per-frame byte reads


def test_ensure_file_bounds_is_fast(tmp_path: Path) -> None:
    """Verify that _ensure_file_bounds does not trigger get_metadata_arrays."""
    run_dir = tmp_path / "bounds_run.pffd"
    run_dir.mkdir()
    path = run_dir / "start_2024.dp_img16.bpp_2.module_1.seqno_0.pff"
    
    create_dummy_pff(path, n_frames=100)
    seq = PFFSequence([path])
    
    with patch.object(seq, 'get_metadata_arrays') as mock_gma:
        seq._ensure_file_bounds()
        assert mock_gma.call_count == 0
        assert seq._file_ts_bounds[0] is not None
        assert seq._file_ts_bounds[0][1] > seq._file_ts_bounds[0][0]

