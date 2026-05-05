import pytest
from pathlib import Path
import numpy as np
from pypff.io2 import PanosetiRun, PFFSequence, hkpff, qconfig
from pypff.models import QuaboHeader, ModuleHeader, DataConfig, ObsConfig

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"

@pytest.fixture
def example_run():
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data not found.")
    return PanosetiRun(EXAMPLE_DATA_DIR)

def test_panoseti_run_comprehensive(example_run):
    run = example_run
    
    # 1. Product Discovery
    products = run.list_products()
    assert len(products) >= 2
    assert "dp_ph256.bpp_2.module_254" in products
    assert "dp_img16.bpp_2.module_1" in products
    
    # 2. Config Loading
    assert "obs_config" in run.configs
    assert "data_config" in run.configs
    assert run.configs["obs_config"]["name"] == "UCB_lab"
    
    # 3. Get Product
    seq = run.get_product("dp_ph256.bpp_2.module_254")
    assert isinstance(seq, PFFSequence)
    assert len(seq) > 0

def test_pff_sequence_comprehensive():
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    if not ph256_file.exists():
        pytest.skip("PH256 test data not found.")
        
    seq = PFFSequence([ph256_file])
    
    # 1. Structure Analysis
    assert seq.frame_config is not None
    assert seq.frame_config.image_shape == (16, 16)
    assert seq.frame_config.dtype == np.int16
    assert seq.frame_config.bytes_per_pixel == 2
    
    # 2. Frame Retrieval
    header, img = seq.get_frame(0)
    assert isinstance(header, QuaboHeader)
    assert img.shape == (16, 16)
    assert img.dtype == np.int16
    
    # 3. Virtual Array (get_image_array)
    # Full read
    all_imgs = seq.get_image_array()
    assert all_imgs.shape == (len(seq), 16, 16)
    
    # Slice read
    slice_count = min(10, len(seq))
    slice_imgs = seq.get_image_array(start=0, count=slice_count)
    assert slice_imgs.shape == (slice_count, 16, 16)
    np.testing.assert_array_equal(slice_imgs, all_imgs[:slice_count])
    
    # Out of bounds
    with pytest.raises(IndexError):
        seq.get_frame(len(seq))
        
    # Close
    seq.close()
    assert len(seq._open_mmaps) == 0

def test_pff_sequence_module_mode():
    img16_file = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    if not img16_file.exists():
        pytest.skip("IMG16 test data not found.")
        
    seq = PFFSequence([img16_file])
    header, img = seq.get_frame(0)
    
    # Module mode headers use ModuleHeader
    assert isinstance(header, ModuleHeader)
    assert img.shape == (32, 32)
    assert img.dtype == np.uint16
    
    # Timing reconciliation
    assert header.timestamp_ns > 0
    assert header.quabo_0.timestamp_ns == header.timestamp_ns

def test_hkpff_modern_comprehensive():
    hk_file = EXAMPLE_DATA_DIR / "hk.pff"
    if not hk_file.exists():
        pytest.skip("HK test data not found.")
        
    hk = hkpff(str(hk_file))
    info = hk.readhk()
    
    assert "QUABO_1019" in info
    assert "WPS" in info
    assert "GPSPRIM" in info
    
    # Check data integrity (TEMP mapping)
    # Legacy: TEMP1 -> DET_TEMP, TEMP2 -> FPGA_TEMP
    assert "DET_TEMP" in info["QUABO_1019"]
    assert "FPGA_TEMP" in info["QUABO_1019"]
    assert isinstance(info["QUABO_1019"]["DET_TEMP"][0], float)

def test_qconfig_modern_comprehensive():
    pattern = str(EXAMPLE_DATA_DIR / "*.json")
    conf = qconfig(pattern).config
    
    assert "obs_config" in conf
    assert "data_config" in conf
    assert "daq_config" in conf
    assert "quabo_uids" in conf
    
    # Validate with models
    obs = ObsConfig(**conf["obs_config"])
    assert obs.name == "UCB_lab"
    
    data = DataConfig(**conf["data_config"])
    assert data.run_type == "pe-steps-ph8"

def test_pff_sequence_multi_file():
    # Simulate multi-file sequence by providing the same file twice
    # (PFFSequence should treat them as seqno_0 and potentially fail extraction 
    # if it can't distinguish, but our extract_seqno helper works on filename)
    # Actually, let's create symlinks or temporary copies to test ordering.
    import tempfile
    import shutil
    
    source_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    if not source_file.exists():
        pytest.skip("Source file not found.")
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        f1 = tmp_path / "test.dp_ph256.module_1.seqno_0.pff"
        f2 = tmp_path / "test.dp_ph256.module_1.seqno_1.pff"
        shutil.copy(source_file, f1)
        shutil.copy(source_file, f2)
        
        seq = PFFSequence([f1, f2])
        
        # Length should be double
        source_len = PFFSequence([source_file])._total_frames
        assert len(seq) == source_len * 2
        
        # Test locating frames across files
        # Frame 0 is in file 0
        file_idx, local_idx = seq._locate_frame(0)
        assert file_idx == 0
        assert local_idx == 0
        
        # Frame source_len is in file 1
        file_idx, local_idx = seq._locate_frame(source_len)
        assert file_idx == 1
        assert local_idx == 0
        
        # Frame source_len - 1 is in file 0
        file_idx, local_idx = seq._locate_frame(source_len - 1)
        assert file_idx == 0
        assert local_idx == source_len - 1
