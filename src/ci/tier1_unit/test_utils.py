import pytest
from pathlib import Path
from pypff.utils import (
    get_coarse_time_ns,
    get_precise_time_ns,
    parse_filename,
    extract_seqno,
)

def test_coarse_time():
    # 1687995222.5777988
    tv_sec = 1687995222
    tv_usec = 577798
    expected = 1687995222577798000
    assert get_coarse_time_ns(tv_sec, tv_usec) == expected

def test_precise_time_no_tai():
    # NTP ahead of GPS (GPS near 1s, NTP near 0s)
    # ntp_nsec = 30ms, pkt_nsec = 970ms -> final_sec = tv_sec - 1
    tv_sec = 1000
    tv_usec = 30_000 # 30ms
    pkt_nsec = 970_000_000 # 970ms
    # tv_nsec_equiv = 30,000,000
    # diff = 30,000,000 - 970,000,000 = -940,000,000
    # diff < -500,000,000 -> final_sec = 999
    expected = 999 * 1_000_000_000 + 970_000_000
    assert get_precise_time_ns(tv_sec, tv_usec, pkt_nsec) == expected

def test_precise_time_with_tai():
    # TAI = UTC + 37
    # d = (tv_sec - pkt_tai + 37) % 1024
    # Case: In sync (d=0)
    tv_sec = 1687995222
    pkt_tai = (tv_sec + 37) % 1024
    pkt_nsec = 123456789
    assert get_precise_time_ns(tv_sec, 0, pkt_nsec, pkt_tai) == tv_sec * 1_000_000_000 + pkt_nsec

def test_parse_filename():
    fname = "start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff"
    meta = parse_filename(fname)
    assert meta['dp'] == 'ph256'
    assert meta['bpp'] == 2
    assert meta['module'] == 254
    assert meta['seqno'] == 0

def test_extract_seqno():
    path = Path("some/dir/file.seqno_5.pff")
    assert extract_seqno(path) == 5
    assert extract_seqno(Path("no_seqno.pff")) == 0
