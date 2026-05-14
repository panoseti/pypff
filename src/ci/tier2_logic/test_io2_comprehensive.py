from pathlib import Path

import numpy as np
import pytest

from pypff.io2 import PanosetiRun, PFFSequence, hkpff
from pypff.models import DataConfig, ModuleHeader, ObsConfig, QuaboHeader

EXAMPLE_DATA_DIR = Path(__file__).parents[3] / "example" / "example-data"


@pytest.fixture
def example_run() -> PanosetiRun:
    if not EXAMPLE_DATA_DIR.exists():
        pytest.skip("Example data not found.")
    return PanosetiRun(EXAMPLE_DATA_DIR)


def test_panoseti_run_comprehensive(example_run: PanosetiRun) -> None:
    run = example_run

    products = run.list_products()
    assert len(products) >= 2
    assert "dp_ph256.bpp_2.module_254" in products
    assert "dp_img16.bpp_2.module_1" in products

    assert "obs_config" in run.configs
    assert "data_config" in run.configs
    assert run.configs["obs_config"].name == "UCB_lab"

    seq = run.get_product("dp_ph256.bpp_2.module_254")
    assert isinstance(seq, PFFSequence)
    assert len(seq) > 0


def test_pff_sequence_comprehensive() -> None:
    ph256_file = EXAMPLE_DATA_DIR / "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    if not ph256_file.exists():
        pytest.skip("PH256 test data not found.")

    seq = PFFSequence([ph256_file])

    # Structure
    assert seq.frame_config is not None
    assert seq.frame_config.image_shape == (16, 16)
    assert seq.frame_config.dtype == np.int16
    assert seq.frame_config.bytes_per_pixel == 2

    # get_frame returns (dict, ndarray_view) by default
    header, img = seq.get_frame(0)
    assert isinstance(header, dict)
    assert "pkt_num" in header
    assert img.shape == (16, 16)
    assert img.dtype == np.int16

    # get_frame_validated returns Pydantic object
    header_v, img_v = seq.get_frame_validated(0)
    assert isinstance(header_v, QuaboHeader)
    assert img_v.shape == (16, 16)

    # read_images_range
    all_imgs = seq.read_images_range(0)
    assert all_imgs.shape == (len(seq), 16, 16)

    slice_count = min(10, len(seq))
    slice_imgs = seq.read_images_range(0, slice_count)
    assert slice_imgs.shape == (slice_count, 16, 16)
    np.testing.assert_array_equal(slice_imgs, all_imgs[:slice_count])

    # Out of bounds
    with pytest.raises(IndexError):
        seq.get_frame(len(seq))

    # Timestamp API
    t0 = seq.timestamp_at(0)
    assert t0 > 0

    # timestamps() returns full int64 array
    ts_all = seq.timestamps()
    assert ts_all.dtype == np.int64
    assert len(ts_all) == len(seq)
    assert ts_all[0] == t0

    # Seek time
    idx = seq.seek_time(t0 + 1)
    assert idx == 0 or idx == 1

    # Context manager closes handles
    with seq:
        pass
    assert len(seq._lru) == 0


def test_pff_sequence_module_mode() -> None:
    img16_file = EXAMPLE_DATA_DIR / "start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff"
    if not img16_file.exists():
        pytest.skip("IMG16 test data not found.")

    seq = PFFSequence([img16_file])

    # get_frame returns dict
    header_d, img = seq.get_frame(0)
    assert isinstance(header_d, dict)
    assert "quabo_0" in header_d
    assert img.shape == (32, 32)
    assert img.dtype == np.uint16

    # get_frame_validated returns Pydantic ModuleHeader
    header_v, _ = seq.get_frame_validated(0)
    assert isinstance(header_v, ModuleHeader)
    assert header_v.timestamp_ns > 0
    assert header_v.quabo_0.timestamp_ns == header_v.timestamp_ns


def test_hkpff_modern_comprehensive() -> None:
    hk_file = EXAMPLE_DATA_DIR / "hk.pff"
    if not hk_file.exists():
        pytest.skip("HK test data not found.")

    hk = hkpff(str(hk_file))
    info = hk.readhk()

    assert "QUABO_1019" in info
    assert "WPS" in info
    assert "GPSPRIM" in info

    # TEMP1 -> DET_TEMP, TEMP2 -> FPGA_TEMP remapping
    assert "DET_TEMP" in info["QUABO_1019"]
    assert "FPGA_TEMP" in info["QUABO_1019"]

    # Values are now numpy arrays
    det_temp = info["QUABO_1019"]["DET_TEMP"]
    assert isinstance(det_temp, np.ndarray)
    assert len(det_temp) > 0
    assert np.issubdtype(det_temp.dtype, np.floating) or np.issubdtype(det_temp.dtype, np.integer)


def test_qconfig_modern_comprehensive() -> None:
    run = PanosetiRun(EXAMPLE_DATA_DIR)
    conf = run.configs

    assert "obs_config" in conf
    assert "data_config" in conf
    assert "daq_config" in conf

    from pypff.models import QuaboConfig
    assert isinstance(conf["obs_config"], ObsConfig)
    assert isinstance(conf["data_config"], DataConfig)

    assert conf["obs_config"].name == "UCB_lab"
    assert conf["data_config"].run_type == "pe-steps-ph8"

    q_key = "quabo_config_192.168.3.248"
    if q_key in conf:
        assert isinstance(conf[q_key], QuaboConfig)
        assert conf[q_key].OTABG_ON == [1, 1, 1, 1]


def test_pff_sequence_multi_file() -> None:
    import shutil
    import tempfile

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
        source_len = PFFSequence([source_file])._total_frames
        assert len(seq) == source_len * 2

        file_idx, local_idx = seq._locate_frame(0)
        assert file_idx == 0 and local_idx == 0

        file_idx, local_idx = seq._locate_frame(source_len)
        assert file_idx == 1 and local_idx == 0

        file_idx, local_idx = seq._locate_frame(source_len - 1)
        assert file_idx == 0 and local_idx == source_len - 1
