import numpy as np
import pytest
import concurrent.futures
import pickle
from pathlib import Path
from pypff.io2 import PFFSequence, PanosetiRun

def create_dummy_pff(path: Path, n_frames: int = 10, is_module: bool = False, start_idx: int = 0):
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
def dummy_run(tmp_path):
    run_dir = tmp_path / "test_run.pffd"
    run_dir.mkdir()
    create_dummy_pff(run_dir / "start_2024-01-01T00:00:00Z.dp_img16.bpp_2.module_1.seqno_0.pff", n_frames=10, start_idx=0)
    create_dummy_pff(run_dir / "start_2024-01-01T00:00:10Z.dp_img16.bpp_2.module_1.seqno_1.pff", n_frames=10, start_idx=10)
    (run_dir / "obs_config.json").write_text('{"name": "test_obs", "domes": []}')
    (run_dir / "data_config.json").write_text('{"run_type": "test", "image": {"integration_time_usec": 1000, "pe_threshold": 1.0, "quabo_sample_size": 16}}')
    return run_dir

def test_pffsequence_slicing(dummy_run):
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

def read_frame_sum(s, i):
    _, img = s.get_frame(i)
    return img.sum()

def test_pffsequence_multiprocessing(dummy_run):
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(read_frame_sum, seq, i) for i in range(len(seq))]
        results = [f.result() for f in futures]
    assert len(results) == 20

def test_pffsequence_timing_and_seek(dummy_run):
    files = sorted(list(dummy_run.glob("*.pff")))
    seq = PFFSequence(files)
    t0 = seq.get_frame_time(0)
    t1 = seq.get_frame_time(1)
    assert t1 > t0
    idx = seq.seek_time(t0 + 500)
    assert idx in [0, 1]

def test_pffsequence_stress_many_files(tmp_path):
    run_dir = tmp_path / "stress_run.pffd"
    run_dir.mkdir()
    n_files, frames_per_file = 50, 2
    for i in range(n_files):
        path = run_dir / f"start_2024.dp_img16.bpp_2.module_1.seqno_{i}.pff"
        create_dummy_pff(path, n_frames=frames_per_file, start_idx=i*frames_per_file)
    files = sorted(list(run_dir.glob("*.pff")), key=lambda x: int(x.name.split('seqno_')[1].split('.')[0]))
    seq = PFFSequence(files)
    indices = np.random.choice(len(seq), 20, replace=False)
    imgs = seq.get_image_array(indices=indices)
    assert imgs.shape == (20, 32, 32)
