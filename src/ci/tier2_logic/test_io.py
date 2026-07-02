from pathlib import Path

import numpy as np
import pytest

from pypff.io2 import PanosetiRun, PFFSequence, hkpff

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"

def test_panoseti_run_scan() -> None:
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data directory not found.")
    
    run = PanosetiRun(EXAMPLE_DATA_DIR)
    products = run.list_products()
    
    assert "dp_ph256.bpp_2.module_254" in products
    assert "dp_img16.bpp_2.module_1" in products

def test_pff_sequence_ph256() -> None:
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data directory not found.")
    
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    seq = PFFSequence([ph256_file])
    
    assert len(seq) > 0
    header, img = seq.get_frame(0)
    
    assert img.shape == (16, 16)
    assert img.dtype == np.int16
    assert "pkt_num" in header  # get_frame returns a dict by default

def test_pff_sequence_img16() -> None:
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data directory not found.")
    
    img16_file = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    seq = PFFSequence([img16_file])
    
    assert len(seq) > 0
    _header, img = seq.get_frame(0)
    
    assert img.shape == (32, 32)
    assert img.dtype == np.uint16

def test_hkpff_read() -> None:
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data directory not found.")
    
    hk_file = EXAMPLE_DATA_DIR / "hk.pff"
    hk = hkpff(str(hk_file))
    info = hk.readhk()
    
    assert "QUABO_1019" in info
    assert "DET_TEMP" in info["QUABO_1019"]
    assert len(info["QUABO_1019"]["DET_TEMP"]) > 0

def test_get_image_array() -> None:
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data directory not found.")
        
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    seq = PFFSequence([ph256_file])
    
    # Read first 5 frames as a stacked array
    arr = seq.read_images_range(0, 5)
    assert arr.shape == (5, 16, 16)
    assert arr.dtype == np.int16
