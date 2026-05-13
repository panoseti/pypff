import pytest
from pathlib import Path
import numpy as np
from pypff.io import datapff as datapff_legacy
from pypff.io2 import PFFSequence

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"

def test_parsing_methods_ph256():
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    if not ph256_file.exists():
        pytest.skip("PH256 test data not found.")
        
    # 1. Legacy IO (vectorized read)
    dp_leg = datapff_legacy(str(ph256_file))
    leg_data, leg_meta = dp_leg.readpff(metadata=True)
    # Reshape legacy data to (-1, H, W) for comparison
    leg_data_reshaped = leg_data.reshape(-1, 16, 16)
    
    # 2. Modern IO2 Vectorized
    seq = PFFSequence([ph256_file])
    mod_vec_data = seq.read_images_range(0)
    mod_vec_meta = seq.get_all_metadata()
    
    # 3. Modern IO2 Naive Iteration
    naive_data = []
    naive_pkt_nums = []
    for i in range(len(seq)):
        header, img = seq.get_frame(i)
        naive_data.append(img)
        # Handle dict or model safely
        if hasattr(header, 'pkt_num'):
            naive_pkt_nums.append(header.pkt_num)
        else:
            naive_pkt_nums.append(header['pkt_num'])
    
    naive_data_arr = np.array(naive_data)
    naive_pkt_nums_arr = np.array(naive_pkt_nums)
    
    # --- Assertions on Image Data ---
    np.testing.assert_array_equal(leg_data_reshaped, mod_vec_data, err_msg="Legacy data mismatch with IO2 Vectorized data")
    np.testing.assert_array_equal(naive_data_arr, mod_vec_data, err_msg="IO2 Naive data mismatch with IO2 Vectorized data")
    
    # --- Assertions on Metadata ---
    # Legacy meta returns bytes/strings, need to cast
    leg_pkt_nums = np.array([int(x) for x in leg_meta['pkt_num']])
    mod_vec_pkt_nums = mod_vec_meta['pkt_num']
    
    np.testing.assert_array_equal(leg_pkt_nums, mod_vec_pkt_nums, err_msg="Legacy metadata mismatch with IO2 Vectorized metadata")
    np.testing.assert_array_equal(naive_pkt_nums_arr, mod_vec_pkt_nums, err_msg="IO2 Naive metadata mismatch with IO2 Vectorized metadata")

def test_parsing_methods_img16():
    img16_file = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    if not img16_file.exists():
        pytest.skip("IMG16 test data not found.")
        
    # 1. Legacy IO (vectorized read)
    dp_leg = datapff_legacy(str(img16_file))
    leg_data, leg_meta = dp_leg.readpff(metadata=True)
    leg_data_reshaped = leg_data.reshape(-1, 32, 32)
    
    # 2. Modern IO2 Vectorized
    seq = PFFSequence([img16_file])
    mod_vec_data = seq.read_images_range(0)
    mod_vec_meta = seq.get_all_metadata()
    
    # 3. Modern IO2 Naive Iteration
    naive_data = []
    naive_pkt_nums = []
    for i in range(len(seq)):
        header, img = seq.get_frame(i)
        naive_data.append(img)
        if hasattr(header, 'quabo_0'):
            naive_pkt_nums.append(header.quabo_0.pkt_num)
        else:
            naive_pkt_nums.append(header['quabo_0']['pkt_num'])
            
    naive_data_arr = np.array(naive_data)
    naive_pkt_nums_arr = np.array(naive_pkt_nums)
    
    # --- Assertions on Image Data ---
    np.testing.assert_array_equal(leg_data_reshaped, mod_vec_data, err_msg="Legacy data mismatch with IO2 Vectorized data")
    np.testing.assert_array_equal(naive_data_arr, mod_vec_data, err_msg="IO2 Naive data mismatch with IO2 Vectorized data")
    
    # --- Assertions on Metadata ---
    # Legacy module meta is nested under quabo_0
    leg_pkt_nums = np.array([int(x) for x in leg_meta['quabo_0']['pkt_num']])
    mod_vec_pkt_nums = mod_vec_meta['quabo_0']['pkt_num']
    
    np.testing.assert_array_equal(leg_pkt_nums, mod_vec_pkt_nums, err_msg="Legacy metadata mismatch with IO2 Vectorized metadata")
    np.testing.assert_array_equal(naive_pkt_nums_arr, mod_vec_pkt_nums, err_msg="IO2 Naive metadata mismatch with IO2 Vectorized metadata")
