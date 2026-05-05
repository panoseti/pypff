import datetime
from pathlib import Path
from typing import Any
import numpy as np

def get_coarse_time_ns(tv_sec: int, tv_usec: int) -> int:
    """
    Calculates a coarse-resolution timestamp in NANOSECONDS
    using only the system clock.
    """
    return (tv_sec * 1_000_000_000) + (tv_usec * 1_000)

def get_precise_time_ns(tv_sec: int, tv_usec: int, pkt_nsec: int, pkt_tai: int = 0) -> int:
    """
    Calculates precise timestamp in NANOSECONDS using integer arithmetic.
    Logic ported from panoseti_interface.py.
    """
    final_sec = tv_sec

    if pkt_tai != 0:
        # Reconciliation based on quabo's 10-bit TAI counter
        d = (tv_sec - pkt_tai + 37) % 1024
        if d == 1:
            final_sec = tv_sec - 1
        elif d == 1023:
            final_sec = tv_sec + 1
    else:
        # Sub-second reconciliation (500ms threshold)
        tv_nsec_equiv = tv_usec * 1000
        diff = tv_nsec_equiv - pkt_nsec

        if diff > 500_000_000:
            final_sec = tv_sec + 1
        elif diff < -500_000_000:
            final_sec = tv_sec - 1

    return (final_sec * 1_000_000_000) + pkt_nsec

def parse_filename(fname: str) -> dict[str, str | int]:
    """Parses PFF filename attributes."""
    meta: dict[str, str | int] = {}
    clean_name = fname.split('.pff')[0]
    parts = clean_name.split('.')
    for part in parts:
        if '_' in part:
            k, v_str = part.split('_', 1)
            if v_str.isdigit():
                meta[k] = int(v_str)
            else:
                meta[k] = v_str
    return meta

def extract_seqno(filepath: Path) -> int:
    """Extracts sequence number from PFF filename."""
    for part in filepath.name.split('.'):
        if part.startswith('seqno_'):
            try:
                return int(part.split('_')[1])
            except ValueError:
                pass
    return 0
