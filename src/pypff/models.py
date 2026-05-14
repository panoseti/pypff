"""
models.py

Centralized Pydantic models for validating PANOSETI configuration files,
PFF headers, and calibration data.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    IPvAnyAddress,
    ValidationError,
    field_validator,
    model_validator,
)

from .utils import get_precise_time_ns

# Global restrictions
MAX_RUN_TYPE_LENGTH = 14
INVALID_RUN_TYPE_CHARS = [".", "_", " "]

MIN_PULSE_HEIGHT_PE_THRESHOLD = 2.0
MIN_MOVIE_MODE_PE_THRESHOLD = 1.0


# ---------------------------
# ---Shared & Base Models ---
# ---------------------------

class BaseStrictModel(BaseModel):
    """Disallows extra fields and mutations to catch typos in configuration keys."""
    model_config = ConfigDict(extra='forbid', frozen=True)


# ---------------------------
# --- PFF Metadata Models ---
# ---------------------------

class PFFHeader(BaseStrictModel):
    """Standardized PanoSETI File Format (PFF) JSON header."""
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
    """PFF header specific to a single Quabo's data."""
    quabo_num: int


class ModuleHeader(BaseStrictModel):
    """Hierarchical PFF header containing telemetry for a full 4-Quabo module."""
    quabo_0: PFFHeader
    quabo_1: PFFHeader
    quabo_2: PFFHeader
    quabo_3: PFFHeader

    @property
    def timestamp_ns(self) -> int:
        for i in range(4):
            q = getattr(self, f"quabo_{i}")
            if q.tv_sec != 0:
                return q.timestamp_ns
        return self.quabo_0.timestamp_ns


class FrameConfig(BaseStrictModel):
    header_size: int
    payload_size: int
    frame_size: int
    image_shape: tuple[int, int]
    dtype_str: str
    bytes_per_pixel: int
    format_name: str

    @functools.cached_property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)


# --------------------------
# --- Data Config Models ---
# --------------------------

class AnyTriggerConfig(BaseStrictModel):
    """Configuration for 'any_trigger' mode in pulse height acquisition."""
    group_ph_frames: int = Field(0, description="If set to 1, hashpipe will group 4 packets from 4 quabos.")


class PulseHeightMode(BaseStrictModel):
    """Parameters for Pulse Height (PH) data acquisition."""
    pe_threshold: float = Field(..., description="Pulse height threshold in photoelectrons")
    any_trigger: AnyTriggerConfig | None = None
    two_pixel_trigger: int = Field(0, description="If set to 1, 2 pixel trigger mode will be enabled.")
    three_pixel_trigger: int = Field(0, description="If set to 1, 3 pixel trigger mode will be enabled.")


class ImageMode(BaseStrictModel):
    """Parameters for Image (Movie) mode data acquisition."""
    integration_time_usec: int = Field(..., ge=20, description="Integration time in microseconds")
    pe_threshold: float = Field(..., description="Image mode threshold in photoelectrons")
    quabo_sample_size: Literal[8, 16] = Field(..., description="Size of the sample")

    @field_validator('integration_time_usec')
    def check_integration_time_divisor(cls, v: int) -> int:
        if 1_000_000 % v != 0:
            raise ValueError(f"integration_time_usec ({v}) must evenly divide 1,000,000 usec.")
        return v


class LongPulseMode(BaseStrictModel):
    """Parameters for long-pulse event detection."""
    octaves: int
    threshold_sigma: list[float]


class FlashParams(BaseStrictModel):
    """Controls for the onboard LED flash system."""
    rate: int = Field(..., ge=0, le=7, description="3-bit value controlling flash rate (0-7)")
    level: int = Field(..., ge=0, le=31, description="Controls DC supply level (0-31)")
    width: int = Field(..., ge=0, le=15, description="Controls pulse width (0-15)")


class StimParams(BaseStrictModel):
    """Controls for the electronic stimulus (test pulse) system."""
    rate: int = Field(..., ge=0, le=7, description="Rate from 190 to 24,400 Hz")
    level: int = Field(..., ge=0, le=255)
    mask: list[bool] = Field(..., max_length=4, min_length=4)


class InterleaveState(BaseStrictModel):
    """A single state in an interleaved observing sequence."""
    state_name: str
    duration_seconds: float = Field(..., gt=0.01)
    movie_mode_config: str | None = None
    pulse_height_mode_config: str | None = None

    @model_validator(mode='after')
    def check_at_least_one_mode(self) -> InterleaveState:
        if not self.movie_mode_config and not self.pulse_height_mode_config:
            raise ValueError(f"State '{self.state_name}' must have at least one valid mode (movie or pulse_height).")
        return self


class InterleaveConfig(BaseStrictModel):
    """Full configuration for cyclical interleaved observing."""
    enable: bool = Field(False)
    states: list[InterleaveState] = Field([])


class DataConfig(BaseModel):
    """Science and engineering acquisition parameters (data_config.json)."""
    model_config = ConfigDict(extra='allow')

    run_type: str = Field(..., max_length=MAX_RUN_TYPE_LENGTH)
    detector_overvoltage: Literal[2, 3] | None = None
    gain: int | None = None
    max_file_size_mb: int | None = Field(None, gt=0)
    image: ImageMode | None = None
    pulse_height: PulseHeightMode | None = None
    interleave: InterleaveConfig | None = None
    stim_params: StimParams | None = None
    flash_params: FlashParams | None = None
    xfr_bwlimit: int | None = Field(None, gt=0)

    @field_validator("run_type")
    def validate_run_type(cls, v: str) -> str:
        if any(ch in INVALID_RUN_TYPE_CHARS for ch in v):
            raise ValueError(
                f"Invalid run_type: '{v}' contains at least one invalid character: {INVALID_RUN_TYPE_CHARS}")
        return v

    @model_validator(mode='after')
    def validate_dynamic_modes(self) -> DataConfig:
        if not self.model_extra:
            return self
        for key, val in self.model_extra.items():
            if key.startswith('image_'):
                try:
                    ImageMode(**val)
                except ValidationError as e:
                    raise ValueError(f"Invalid fields in dynamic mode '{key}': {e}") from e
            elif key.startswith('pulse_height_'):
                try:
                    PulseHeightMode(**val)
                except ValidationError as e:
                    raise ValueError(f"Invalid fields in dynamic mode '{key}': {e}") from e
        return self


# -------------------------
# --- Obs Config Models ---
# -------------------------

class WpsConfig(BaseStrictModel):
    """Configuration for a Web Power Switch (WPS) unit."""
    url: str
    quabo_socket: int


class ObsModuleConfig(BaseModel):
    """Configuration and state for a single observatory module."""
    model_config = ConfigDict(extra='allow')
    mobo_serialno: str
    quabo_version: str | list[str]
    ip_addr: IPvAnyAddress
    wps: str | None = None
    ups: str | None = None
    timing_mode: str | None = Field("wr", pattern="^(wr|gnss)$")
    azimuth: float | None = Field(None, ge=0, le=360)
    elevation: float | None = Field(None, ge=-90, le=90)
    position_angle: float | None = None

    id: int | None = None
    daq_node: Any | None = None


class ObsDomeConfig(BaseStrictModel):
    """Physical configuration for an observatory dome."""
    name: str
    obslat: float = Field(..., ge=-90, le=90)
    obslon: float = Field(..., ge=-180, le=180)
    obsalt: float
    modules: list[ObsModuleConfig]
    num: int | None = None


class ObsConfig(BaseModel):
    """Physical observatory setup and device mapping (obs_config.json)."""
    name: str
    comment: str | None = None
    wr_ip_addr: IPvAnyAddress | None = IPvAnyAddress("192.168.1.254")  # type: ignore
    dome_controller_ip_addr: IPvAnyAddress | None = None
    gps_port: str | None = Field("/dev/ttyUSB0")
    detector_overvoltage: int | None = None
    domes: list[ObsDomeConfig]

    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    def validate_wps_extras(self) -> ObsConfig:
        extra_data = self.model_extra or {}
        for key, val in extra_data.items():
            if key.startswith("wps"):
                try:
                    WpsConfig(**val)
                except Exception as e:
                    raise ValueError(f"Invalid format for '{key}': {e}") from e
        return self


# -------------------------
# --- DAQ Config Models ---
# -------------------------

class PortForwarding(BaseStrictModel):
    """Networking metadata for port-forwarded devices (Gateways)."""
    status: bool = Field(False)
    gw_ip: IPvAnyAddress
    reboot_port: list[int | None] | None = Field(None)
    cmd_port: list[int | None] | None = Field(None)
    port: int | None = None
    grpc_port: int = Field(50051, ge=1, le=65535)


class DaqNode(BaseModel):
    """Configuration for a single remote Data Acquisition (DAQ) node."""
    model_config = ConfigDict(extra='allow')
    username: str
    data_dir: str
    ip_addr: IPvAnyAddress
    module_ids: list[int] | str | int
    bindhost: str | None = Field("0.0.0.0")
    port_forwarding: PortForwarding | None = None

    @field_validator('module_ids', mode='after')
    def parse_module_ids(cls, v: int | str | list[int]) -> list[int]:
        if isinstance(v, list):
            return v
        elif isinstance(v, int):
            return [v]
        elif isinstance(v, str):
            if re.match(r'^\d+\-\d+$', v):
                start, end = map(int, v.split('-'))
                return list(range(start, end + 1))
            elif ',' in v:
                return [int(x.strip()) for x in v.split(',')]
            elif v.isdigit():
                return [int(v)]
        return []


class DaqConfig(BaseStrictModel):
    """DAQ node networking and storage configuration (daq_config.json)."""
    comment: str | None = None
    head_node_data_dir: str
    head_node_ip_addr: IPvAnyAddress
    head_node_container: bool | None = Field(False)
    daq_node_module_limit: int | None = Field(4)
    daq_nodes: list[DaqNode]


# -----------------------------
# --- Network Config Models ---
# -----------------------------

class NetworkModule(BaseStrictModel):
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding


class NetworkDaqNode(BaseStrictModel):
    ip_addr: IPvAnyAddress
    port_forwarding: PortForwarding


class NetworkConfig(BaseStrictModel):
    modules: list[NetworkModule] = Field(default_factory=list)
    daq_nodes: list[NetworkDaqNode] = Field(default_factory=list)


# ----------------------------
# --- Daemon Config Models ---
# ----------------------------

class Daemons(BaseModel):
    model_config = ConfigDict(extra='allow')


class DaemonConfig(BaseStrictModel):
    daemons: Daemons
    permanent_daemons: Daemons | None = None


# ------------------------------
# --- Firmware Config Models ---
# ------------------------------

class FirmwareConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    qfp: str | None = None
    bga: str | None = None
    gold: str | None = None


# --------------------------------
# --- Quabo UIDs Config Models ---
# --------------------------------

class QuaboUidEntry(BaseStrictModel):
    uid: str


class QuaboUidModule(BaseModel):
    model_config = ConfigDict(extra='allow')
    ip_addr: IPvAnyAddress
    quabos: list[QuaboUidEntry] = Field(..., min_length=4, max_length=4)


class QuaboUidDome(BaseStrictModel):
    modules: list[QuaboUidModule]


class QuaboUids(BaseStrictModel):
    domes: list[QuaboUidDome]


# ---------------------------------
# --- Quabo Configuration Models ---
# ---------------------------------

def parse_csv_ints(v: Any) -> list[int]:
    if isinstance(v, str):
        parts = v.split(',')
        res = []
        for p in parts:
            p = p.strip()
            if p.startswith('0x'):
                res.append(int(p, 16))
            else:
                res.append(int(p))
        return res
    return v


class QuaboConfig(BaseModel):
    """Internal register settings for a Quabo board (quabo_config_IP.json)."""
    model_config = ConfigDict(extra='allow')

    DAC1: str | list[int] | None = None
    DAC2: str | list[int] | None = None
    GAIN0: str | list[int] | None = None

    @model_validator(mode='before')
    @classmethod
    def handle_csv_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for k, v in data.items():
            if isinstance(v, str) and ',' in v:
                data[k] = parse_csv_ints(v)
            elif isinstance(v, str) and v.startswith('0x'):
                data[k] = int(v, 16)
            elif isinstance(v, str) and v.lstrip('-').isdigit():
                data[k] = int(v)
        return data


# ---------------------------------
# --- Pulse Height Baseline Models ---
# ---------------------------------

class QuaboPhBaseline(BaseStrictModel):
    """Pulse height baseline calibration for a single Quabo."""
    uid: str
    coefs: list[int] = Field(..., min_length=256, max_length=256)


class PhBaselineConfig(BaseStrictModel):
    """Pulse Height (PH) baseline calibration registry."""
    date: str
    quabos: list[QuaboPhBaseline]
