import concurrent.futures
from pathlib import Path

import numpy as np
import pytest

from pypff.io2 import PanosetiRun


def create_module_pff(path: Path, n_frames: int = 10) -> None:
    header_base = (
        '{"quabo_0": {"pkt_num": %10d, "pkt_tai": 0, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}, '
        '"quabo_1": {"pkt_num": %10d, "pkt_tai": 0, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}, '
        '"quabo_2": {"pkt_num": %10d, "pkt_tai": 0, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}, '
        '"quabo_3": {"pkt_num": %10d, "pkt_tai": 0, "pkt_nsec": %10d, "tv_sec": 1700000000, "tv_usec": 0}}\n\n*'
    )
    payload = np.zeros((32, 32), dtype=np.uint16).tobytes()
    with open(path, 'wb') as f:
        for i in range(n_frames):
            h = header_base % (i, i*1000, i, i*1000, i, i*1000, i, i*1000)
            f.write(h.encode() + payload)

@pytest.fixture
def complex_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "complex_run.pffd"
    run_dir.mkdir()
    create_module_pff(run_dir / "start_2024.dp_img16.bpp_2.module_1.seqno_0.pff", n_frames=20)
    (run_dir / "hk.pff").write_bytes(b'{"QUABO_0": {"TEMP1": "45.5", "TEMP2": "50.1"}}\n\n')
    (run_dir / "quabo_config_192.168.1.10.json").write_text('{"DAC1": "100,200,300,400", "OTABG_ON": "1"}')
    return run_dir

def test_metadata_structuring(complex_run: Path) -> None:
    run = PanosetiRun(complex_run)
    seq = run.get_product("dp_img16.bpp_2.module_1")
    meta = seq.get_all_metadata()
    assert "quabo_0" in meta
    assert meta["quabo_0"]["pkt_num"][5] == 5

def test_housekeeping_parsing(complex_run: Path) -> None:
    run = PanosetiRun(complex_run)
    hk = run.get_hk()
    assert hk["QUABO_0"]["DET_TEMP"] == [45.5]

def process_run_func(r: PanosetiRun) -> int:
    p_name = r.list_products()[0]
    s = r.get_product(p_name)
    return len(s)

def test_multiprocessing_run(complex_run: Path) -> None:
    run = PanosetiRun(complex_run)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future = executor.submit(process_run_func, run)
        assert future.result() == 20
