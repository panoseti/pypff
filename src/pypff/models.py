from pydantic import BaseModel
import numpy as np
from .utils import get_precise_time_ns

class PFFHeader(BaseModel):
    """Base model for all PFF headers."""
    pkt_num: int
    pkt_tai: int
    pkt_nsec: int
    tv_sec: int
    tv_usec: int

    @property
    def timestamp_ns(self) -> int:
        """Returns nanoseconds since epoch as int64 compatible integer."""
        return get_precise_time_ns(self.tv_sec, self.tv_usec, self.pkt_nsec, self.pkt_tai)


class QuaboHeader(PFFHeader):
    quabo_num: int


class ModuleHeader(BaseModel):
    quabo_0: PFFHeader
    quabo_1: PFFHeader
    quabo_2: PFFHeader
    quabo_3: PFFHeader

    @property
    def timestamp_ns(self) -> int:
        # Use first non-zero quabo for timing
        for i in range(4):
            q = getattr(self, f"quabo_{i}")
            if q.tv_sec != 0:
                return q.timestamp_ns
        return self.quabo_0.timestamp_ns


class FrameConfig(BaseModel):
    header_size: int
    payload_size: int
    frame_size: int
    image_shape: tuple[int, int]
    dtype_str: str
    bytes_per_pixel: int
    format_name: str

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)
