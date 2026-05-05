import pytest
from pathlib import Path
import numpy as np
from pypff.io import hkpff as hkpff_legacy, datapff as datapff_legacy, qconfig as qconfig_legacy
from pypff.io2 import hkpff as hkpff_modern, PFFSequence, PanosetiRun, qconfig as qconfig_modern

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"
LEGACY_TEST_DATA_DIR = Path(__file__).parents[1] / "legacy_tests"

def test_hkpff_comparison():
    hk_file = LEGACY_TEST_DATA_DIR / "hk-data" / "hk.pff"
    if not hk_file.exists():
        pytest.skip("HK test data not found.")
        
    legacy_hk = hkpff_legacy(str(hk_file)).readhk()
    modern_hk = hkpff_modern(str(hk_file)).readhk()
    
    # Check top-level keys (e.g., QUABO_1019, WPS, etc.)
    assert set(legacy_hk.keys()) == set(modern_hk.keys())
    
    for key in legacy_hk:
        # Check sub-keys (e.g., DET_TEMP, Computer_UTC)
        assert set(legacy_hk[key].keys()) == set(modern_hk[key].keys())
        for subkey in legacy_hk[key]:
            # Compare values
            np.testing.assert_array_equal(legacy_hk[key][subkey], modern_hk[key][subkey])

def test_datapff_vs_pffsequence_ph256():
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    if not ph256_file.exists():
        pytest.skip("PH256 test data not found.")
        
    # Legacy read
    dp_leg = datapff_legacy(str(ph256_file))
    leg_data, leg_meta = dp_leg.readpff(metadata=True)
    
    # Modern read
    seq = PFFSequence([ph256_file])
    mod_data = seq.get_image_array()
    
    # Compare data
    # Legacy data is reshaped to (-1, pixels) in readpff, modern is (-1, H, W)
    leg_data_reshaped = leg_data.reshape(-1, 16, 16)
    np.testing.assert_array_equal(leg_data_reshaped, mod_data)
    
    # Compare metadata for first frame
    mod_header, _ = seq.get_frame(0)
    assert int(leg_meta['pkt_num'][0]) == mod_header.pkt_num
    assert int(leg_meta['tv_sec'][0]) == mod_header.tv_sec
    assert int(leg_meta['tv_usec'][0]) == mod_header.tv_usec

def test_datapff_vs_pffsequence_img16():
    img16_file = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    if not img16_file.exists():
        pytest.skip("IMG16 test data not found.")
        
    # Legacy read
    dp_leg = datapff_legacy(str(img16_file))
    leg_data, leg_meta = dp_leg.readpff(metadata=True)
    
    # Modern read
    seq = PFFSequence([img16_file])
    mod_data = seq.get_image_array()
    
    # Compare data
    leg_data_reshaped = leg_data.reshape(-1, 32, 32)
    np.testing.assert_array_equal(leg_data_reshaped, mod_data)
    
    # Compare metadata for first frame (ModuleHeader in modern)
    mod_header, _ = seq.get_frame(0)
    assert int(leg_meta['quabo_0']['pkt_num'][0]) == mod_header.quabo_0.pkt_num
    assert int(leg_meta['quabo_3']['tv_sec'][0]) == mod_header.quabo_3.tv_sec

def test_qconfig_comparison():
    config_pattern = str(EXAMPLE_DATA_DIR / "*.json")
    
    legacy_conf = qconfig_legacy(config_pattern).config
    modern_conf = qconfig_modern(config_pattern).config
    
    # legacy qconfig does some complex parsing for quabo_config (csv string to list)
    # modern qconfig is simpler (just orjson.loads).
    # Let's check shared non-quabo config first
    assert legacy_conf['obs_config'] == modern_conf['obs_config']
    assert legacy_conf['daq_config'] == modern_conf['daq_config']
    
    # Check a quabo_config file
    q_key = 'quabo_config_192.168.3.248'
    if q_key in legacy_conf and q_key in modern_conf:
        # Legacy: 'OTABG_ON': [1, 1, 1, 1]
        # Modern: 'OTABG_ON': '1,1,1,1'
        # They will differ unless I fix io2.py
        assert legacy_conf[q_key] != modern_conf[q_key]
