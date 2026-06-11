"""Shared dtype table for PFF header fields."""
from __future__ import annotations

import numpy as np

_HEADER_DTYPES: dict[str, np.dtype[np.generic]] = {
    "quabo_num": np.dtype("uint8"),
    "pkt_num":   np.dtype("uint32"),
    "pkt_tai":   np.dtype("uint16"),
    "pkt_nsec":  np.dtype("uint32"),
    "tv_sec":    np.dtype("int64"),
    "tv_usec":   np.dtype("uint32"),
}
