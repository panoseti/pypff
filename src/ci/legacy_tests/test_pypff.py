from pathlib import Path
from typing import Any

import numpy as np

import pypff

DATA_DIR = Path(__file__).parent

"""
test hk read
"""
expected_hk_results: dict[str, dict[str, Any]] = {
    "WPS": {"Computer_UTC": 1754975261.080837, "POWER": "ON"},

"QUABO_1019": {"Computer_UTC": "1754975259.3758526", "BOARDLOC": "1019", "HVMON0": "-54.18142", "HVMON1": "-54.14238", "HVMON2": "-54.21558", "HVMON3": "-54.21436", "HVIMON0": "0.0002097024", "HVIMON1": "0.0002102739", "HVIMON2": "0.00020939760000000002", "HVIMON3": "0.00021515070000000002", "RAWHVMON": "71.553", "V12MON": "1.18565818", "V18MON": "1.78506642", "V33MON": "3.3041844", "V37MON": "3.726942", "I10MON": "1.206296", "I18MON": "0.3997728", "I33MON": "0.1133622", "TEMP1": "43.0", "TEMP2": "63.853998769609404", "VCCINT": "0.9824066162109375", "VCCAUX": "1.7735595703125", "UID": "0x0000000000000000", "SHUTTER_STATUS": "0", "LIGHT_SENSOR_STATUS": "0", "PCBREV_N": "1", "FWTIME": "0x2cfb9557", "FWVER": "0209", "StartUp": "0", "AGG_STATUS_MSG": "ok,", "AGG_STATUS_LEVEL": "0", "DETR0_CURR": "0.0001011224", "DETR1_CURR": "0.0001017721364729459", "DETR2_CURR": "0.00010074914308617235", "DETR3_CURR": "0.00010650468797595193"},

"QUABO_1018": {"Computer_UTC": "1754975261.464401", "BOARDLOC": "1018", "HVMON0": "-53.61168", "HVMON1": "-54.11066", "HVMON2": "-54.0826", "HVMON3": "-54.117979999999996", "HVIMON0": "0.0001552956", "HVIMON1": "0.00020654010000000002", "HVIMON2": "0.000204216", "HVIMON3": "0.0002019681", "RAWHVMON": "71.53226", "V12MON": "1.1813865", "V18MON": "1.79109254", "V33MON": "3.2714184", "V37MON": "3.717417", "I10MON": "0.834106", "I18MON": "0.35921339999999996", "I33MON": "0.1157814", "TEMP1": "511.75", "TEMP2": "50.98103660412187", "VCCINT": "0.983367919921875", "VCCAUX": "1.7880706787109375", "UID": "0x0000000000000000", "SHUTTER_STATUS": "0", "LIGHT_SENSOR_STATUS": "0", "PCBREV_N": "1", "FWTIME": "0x2cfb9557", "FWVER": "0209", "StartUp": "0", "AGG_STATUS_MSG": "detr_temp:crit,", "AGG_STATUS_LEVEL": "32", "DETR0_CURR": "4.785736352705411e-05", "DETR1_CURR": "9.810190360721445e-05", "DETR2_CURR": "9.58340360721443e-05", "DETR3_CURR": "9.351523426853708e-05"},

"QUABO_1017": {"Computer_UTC": "1754975261.533732", "BOARDLOC": "1017", "HVMON0": "-54.29366", "HVMON1": "-54.28878", "HVMON2": "-54.31074", "HVMON3": "-54.247299999999996", "HVIMON0": "0.0002255901", "HVIMON1": "0.000216408", "HVIMON2": "0.0002143125", "HVIMON3": "0.00021633180000000003", "RAWHVMON": "71.52128", "V12MON": "1.18110045", "V18MON": "1.78834646", "V33MON": "3.2791146", "V37MON": "3.7004244", "I10MON": "0.954954", "I18MON": "0.37115819999999994", "I33MON": "0.11453399999999998", "TEMP1": "43.25", "TEMP2": "65.3150876653338", "VCCINT": "0.9823150634765625", "VCCAUX": "1.7721405029296875", "UID": "0x0000000000000000", "SHUTTER_STATUS": "0", "LIGHT_SENSOR_STATUS": "0", "PCBREV_N": "1", "FWTIME": "0x2cfb9557", "FWVER": "0209", "StartUp": "0", "AGG_STATUS_MSG": "ok,", "AGG_STATUS_LEVEL": "0", "DETR0_CURR": "0.00011678517014028054", "DETR1_CURR": "0.00010761284969939879", "DETR2_CURR": "0.00010547334168336674", "DETR3_CURR": "0.00010761977595190385"},

"QUABO_1016": {"Computer_UTC": "1754975260.989664", "BOARDLOC": "1016", "HVMON0": "-54.10822", "HVMON1": "-54.12774", "HVMON2": "-54.1375", "HVMON3": "-54.12164", "HVIMON0": "0.0002066163", "HVIMON1": "0.00020585430000000002", "HVIMON2": "0.0002127123", "HVIMON3": "0.0002092452", "RAWHVMON": "71.56886", "V12MON": "1.188061", "V18MON": "1.79345722", "V33MON": "3.2993076", "V37MON": "3.71856", "I10MON": "0.829374", "I18MON": "0.35581139999999994", "I33MON": "0.1140804", "TEMP1": "41.25", "TEMP2": "52.18066748692712", "VCCINT": "0.991607666015625", "VCCAUX": "1.78363037109375", "UID": "0x0000000000000000", "SHUTTER_STATUS": "0", "LIGHT_SENSOR_STATUS": "0", "PCBREV_N": "1", "FWTIME": "0x2cfb9557", "FWVER": "0209", "StartUp": "0", "AGG_STATUS_MSG": "ok,", "AGG_STATUS_LEVEL": "0", "DETR0_CURR": "9.818299338677355e-05", "DETR1_CURR": "9.738187515030061e-05", "DETR2_CURR": "0.00010422031603206413", "DETR3_CURR": "0.00010078499959919841"}
}

def test_read_hk() -> None:
    hkpff = pypff.io.hkpff(str(DATA_DIR / 'hk-data/hk.pff'))
    hk_info = hkpff.readhk()
    for k in hk_info:
        for kk in hk_info[k]:
            try:
                if kk == 'DET_TEMP':
                    v: float | str = float(expected_hk_results[k]['TEMP1'])
                elif kk == 'FPGA_TEMP':
                    v = float(expected_hk_results[k]['TEMP2'])
                else:
                    v = float(expected_hk_results[k][kk])
            except (ValueError, KeyError, TypeError):
                v = expected_hk_results[k][kk]
            assert hk_info[k][kk][0] == v

"""
test pff read
"""
expected_pff_metadata: dict[str, Any] = {
    'quabo_num': 0, 
    'pkt_num': 7112,
    'pkt_tai': 715, 
    'pkt_nsec': 934405331, 
    'tv_sec': 1754109119, 
    'tv_usec': 872646
}

expected_pff_data = np.array(
      [ 19,  13,   7,  14,  13,  16,  20,  13,  18,   7,  32,  15,   7,
        16,  16,  25,  13,   9,  15,  20,   4,  24,  12,   7,  13,  28,
        10,   9,   6,   9,  16,  23,  20,  29,  11,  23,  29,  24,   6,
        15,   4,  13,   9,  27,   4,  15,  19,   8,   9,  28,   6,  21,
        10,  14,   7,  25, -13,  17,  11,  21,  20,  21,   0,  18,  18,
        35,   9,  19,  25,  12,  15,  13,  17,  11,   2,  10,  10,  15,
        12,  17,   9,  11,  22,  26,   1,  15,  24,   7,  38,   6,   3,
        13,  19,  18,  15,  23,  16,   9,  16,  22,  16,  14,   9,  24,
        16,  29,  11,   0,  14,  16,   4,  14,  24,  48,  37,  31,  25,
        -1,   6,  31,  20,  11,  12,  13,  26,  17,  18,  14,  35,  35,
        31,  52,  16,  37,  28,  14,  26,  30,  13,  16,  21,  10,  15,
        19,  16,  61,  32,  66,  25,  34,  30,  17,  26,  13,  10,  24,
        12,  16,  21,   8,  21,  40,  34,  61, 141,  86,  69,  17,  20,
        16,  17,   0,  14,  19,  20,  31,  13,  25,  18,  71,  81, 221,
       155,  60,  44,  22,  10,   4,   9,   6,   3,  22,  14,   3,  14,
        52,  55,  59,  83, 122,  44,  20,  17,  14,  13,   9,  15,  22,
        13,  16,  21,  12,  36,  48,  43,  19,  21,  45,  16,  32,  14,
         8,  16,  13,  20,   2,  18,   9,  20,  27,  39,  11,  18,  17,
        -7,  14,  25,  24,  -2,  14,  18,  22,  12,  23,   8,  20,   8,
        13,  24,  17,  17,  12,  18,  10,  16,   8], dtype=np.int16)

def test_read_pff() -> None:
    dpff = pypff.io.datapff(str(DATA_DIR / 'sci-data/start_2025-08-02T04-31-52Z.dp_ph256.bpp_2.module_254.seqno_0.pff'))
    data, metadata = dpff.readpff(metadata=True)
    # check metadata
    for k in metadata:
        try:
            v: int | str = int(expected_pff_metadata[k])
        except (ValueError, KeyError, TypeError):
            v = expected_pff_metadata[k]
        assert metadata[k][0] == v
    # check data
    results = np.equal(data[0], expected_pff_data)
    for r in results:
        assert r

"""
test read configs:
These are json files, so there is no need to test this.
"""
