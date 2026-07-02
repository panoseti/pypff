from pathlib import Path

import numpy as np
import pytest

from pypff.io import datapff as datapff_legacy
from pypff.io import hkpff as hkpff_legacy
from pypff.io import qconfig as qconfig_legacy
from pypff.io2 import PFFSequence
from pypff.io2 import hkpff as hkpff_modern
from pypff.io2 import qconfig as qconfig_modern

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"
LEGACY_TEST_DATA_DIR = Path(__file__).parents[1] / "legacy_tests"


def test_hkpff_comparison() -> None:
    hk_file = LEGACY_TEST_DATA_DIR / "hk-data" / "hk.pff"
    if not hk_file.exists():
        pytest.skip("HK test data not found.")

    legacy_hk = hkpff_legacy(str(hk_file)).readhk()
    modern_hk = hkpff_modern(str(hk_file)).readhk()

    assert set(legacy_hk.keys()) == set(modern_hk.keys())
    for key in legacy_hk:
        assert set(legacy_hk[key].keys()) == set(modern_hk[key].keys())
        for subkey in legacy_hk[key]:
            # legacy returns lists; modern returns numpy arrays — compare as arrays
            np.testing.assert_array_equal(
                np.asarray(legacy_hk[key][subkey]),
                np.asarray(modern_hk[key][subkey]),
            )


def test_datapff_vs_pffsequence_ph256() -> None:
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    if not ph256_file.exists():
        pytest.skip("PH256 test data not found.")

    dp_leg = datapff_legacy(str(ph256_file))
    leg_data, leg_meta = dp_leg.readpff(metadata=True)

    seq = PFFSequence([ph256_file])
    # read_images_range replaces get_image_array()
    mod_data = seq.read_images_range(0)

    leg_data_reshaped = leg_data.reshape(-1, 16, 16)
    np.testing.assert_array_equal(leg_data_reshaped, mod_data)

    # get_frame returns a dict
    mod_header, _ = seq.get_frame(0)
    assert isinstance(mod_header, dict)
    assert int(leg_meta["pkt_num"][0]) == mod_header["pkt_num"]
    assert int(leg_meta["tv_sec"][0]) == mod_header["tv_sec"]
    assert int(leg_meta["tv_usec"][0]) == mod_header["tv_usec"]


def test_datapff_vs_pffsequence_img16() -> None:
    img16_file = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    if not img16_file.exists():
        pytest.skip("IMG16 test data not found.")

    dp_leg = datapff_legacy(str(img16_file))
    leg_data, leg_meta = dp_leg.readpff(metadata=True)

    seq = PFFSequence([img16_file])
    mod_data = seq.read_images_range(0)

    leg_data_reshaped = leg_data.reshape(-1, 32, 32)
    np.testing.assert_array_equal(leg_data_reshaped, mod_data)

    # get_frame returns a dict; module headers have nested quabo_N dicts
    mod_header, _ = seq.get_frame(0)
    assert isinstance(mod_header, dict)
    assert int(leg_meta["quabo_0"]["pkt_num"][0]) == mod_header["quabo_0"]["pkt_num"]
    assert int(leg_meta["quabo_3"]["tv_sec"][0]) == mod_header["quabo_3"]["tv_sec"]


def test_qconfig_comparison() -> None:
    config_pattern = str(EXAMPLE_DATA_DIR / "*.json")

    legacy_conf = qconfig_legacy(config_pattern).config
    modern_conf = qconfig_modern(config_pattern).config

    assert legacy_conf["obs_config"] == modern_conf["obs_config"]
    assert legacy_conf["daq_config"] == modern_conf["daq_config"]

    q_key = "quabo_config_192.168.3.248"
    if q_key in legacy_conf and q_key in modern_conf:
        assert legacy_conf[q_key] == modern_conf[q_key]
