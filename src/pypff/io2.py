import bisect
import mmap
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import orjson
from pydantic import ValidationError

from .utils import (
    parse_filename,
    extract_seqno,
)
from .models import QuaboHeader, ModuleHeader, FrameConfig

# --- Core I/O Classes ---

class PFFSequence:
    """
    Represents a time-ordered sequence of PFF files for a single data product.
    Zero-copy access via mmap.
    """

    def __init__(self, file_paths: Sequence[str | Path]):
        paths = [Path(p) for p in file_paths]
        if not paths:
            raise ValueError("No files provided.")

        self.file_paths = sorted(paths, key=extract_seqno)
        self.name = self.file_paths[0].name.split('.seqno')[0]
        self.meta = parse_filename(self.file_paths[0].name)

        self.header_size: int = 0
        self.frame_config: Optional[FrameConfig] = None
        self._file_frame_counts: list[int] = []
        self._cumulative_frames: list[int] = []
        self._total_frames: int = 0
        self._timestamps: Optional[np.ndarray] = None

        self._open_mmaps: dict[int, mmap.mmap] = {}
        self._open_files: dict[int, Any] = {}

        self._analyze_structure()
        self._index_files()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Explicitly close file handles."""
        for mm in self._open_mmaps.values():
            mm.close()
        for f in self._open_files.values():
            f.close()
        self._open_mmaps.clear()
        self._open_files.clear()

    def _analyze_structure(self) -> None:
        """Determines frame structure and enforces exact PanoSETI shapes and types."""
        sample_file = next((p for p in self.file_paths if p.stat().st_size > 0), None)
        if not sample_file:
            return

        with open(sample_file, 'rb') as f:
            chunk = f.read(4096)
            match = re.search(b'}\n\n\\*', chunk)
            if not match:
                match = re.search(b'\n\n\\*', chunk)

            if not match:
                raise ValueError(f"Invalid PFF format in {sample_file}")

            self.header_size = match.end() - 1
            fmt = str(self.meta.get('dp', 'unknown')).lower()

            shape: tuple[int, int]
            dtype: Any
            bpp: int
            if 'img8' in fmt:
                shape = (32, 32)
                dtype = np.uint8
                bpp = 1
            elif 'img16' in fmt:
                shape = (32, 32)
                dtype = np.uint16
                bpp = 2
            elif 'ph256' in fmt:
                shape = (16, 16)
                dtype = np.int16
                bpp = 2
            elif 'ph1024' in fmt:
                shape = (32, 32)
                dtype = np.int16
                bpp = 2
            else:
                shape = (32, 32)
                dtype = np.int16
                bpp = 2

            payload_size = shape[0] * shape[1] * bpp
            frame_size = self.header_size + 1 + payload_size

            self.frame_config = FrameConfig(
                header_size=self.header_size,
                payload_size=payload_size,
                frame_size=frame_size,
                image_shape=shape,
                dtype_str=np.dtype(dtype).name,
                bytes_per_pixel=bpp,
                format_name=fmt
            )

    def _index_files(self) -> None:
        total = 0
        if self.frame_config:
            for p in self.file_paths:
                size = p.stat().st_size
                n = size // self.frame_config.frame_size if size > 0 else 0
                self._file_frame_counts.append(n)
                self._cumulative_frames.append(total + n)
                total += n
        self._total_frames = total

    def __len__(self) -> int:
        return self._total_frames

    def _get_mmap(self, file_idx: int) -> Optional[mmap.mmap]:
        if file_idx in self._open_mmaps:
            return self._open_mmaps[file_idx]

        filepath = self.file_paths[file_idx]
        f = open(filepath, 'rb')
        try:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            self._open_files[file_idx] = f
            self._open_mmaps[file_idx] = mm
            return mm
        except ValueError:
            return None

    def _locate_frame(self, idx: int) -> tuple[int, int]:
        if not (0 <= idx < self._total_frames):
            raise IndexError(f"Frame index {idx} out of range")
        file_idx = bisect.bisect_right(self._cumulative_frames, idx)
        prev_limit = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - prev_limit
        return file_idx, local_idx

    def get_frame(self, idx: int) -> tuple[QuaboHeader | ModuleHeader | dict[str, Any], np.ndarray]:
        if not self.frame_config:
            raise RuntimeError("Frame configuration not analyzed.")

        file_idx, local_idx = self._locate_frame(idx)
        conf = self.frame_config
        mm = self._get_mmap(file_idx)
        if not mm:
            raise RuntimeError(f"Failed to access mmap for file {file_idx}")

        offset = local_idx * conf.frame_size
        header_end = offset + conf.header_size
        header_bytes = mm[offset:header_end]
        header_dict = orjson.loads(header_bytes)

        header_obj: QuaboHeader | ModuleHeader | dict[str, Any]
        try:
            if 'quabo_0' in header_dict:
                header_obj = ModuleHeader(**header_dict)
            elif 'quabo_num' in header_dict:
                header_obj = QuaboHeader(**header_dict)
            else:
                header_obj = header_dict
        except ValidationError:
            header_obj = header_dict

        img_start = header_end + 1
        img_end = img_start + conf.payload_size
        img = np.frombuffer(mm[img_start:img_end], dtype=conf.dtype).reshape(conf.image_shape)

        return header_obj, img

    def get_image_array(self, start: int = 0, count: Optional[int] = None) -> np.ndarray:
        if not self.frame_config:
            return np.empty(0)
        if count is None:
            count = self._total_frames - start
        count = min(count, self._total_frames - start)
        if count <= 0:
            return np.empty((0, *self.frame_config.image_shape), dtype=self.frame_config.dtype)

        conf = self.frame_config
        dtype = conf.dtype
        itemsize = dtype.itemsize

        chunks = []
        frames_collected = 0
        current_global = start

        while frames_collected < count:
            file_idx, local_start = self._locate_frame(current_global)
            file_total = self._file_frame_counts[file_idx]
            to_read = min(count - frames_collected, file_total - local_start)

            if to_read > 0:
                mm = self._get_mmap(file_idx)
                if not mm:
                    raise RuntimeError("Failed to read mmap block.")

                start_offset = (local_start * conf.frame_size) + conf.header_size + 1
                can_use_strided = (conf.frame_size % itemsize == 0) and (start_offset % itemsize == 0)

                if can_use_strided:
                    strides = (conf.frame_size, conf.image_shape[1] * itemsize, itemsize)
                    chunk = np.ndarray(shape=(to_read, *conf.image_shape), dtype=dtype, buffer=mm, offset=start_offset, strides=strides)
                    chunks.append(chunk)
                else:
                    temp = np.empty((to_read, *conf.image_shape), dtype=dtype)
                    cursor = start_offset
                    for i in range(to_read):
                        end = cursor + conf.payload_size
                        temp[i] = np.frombuffer(mm[cursor:end], dtype=dtype).reshape(conf.image_shape)
                        cursor += conf.frame_size
                    chunks.append(temp)

            frames_collected += to_read
            current_global += to_read

        if len(chunks) == 1:
            return chunks[0].copy() if chunks[0].base is not None else chunks[0]
        return np.concatenate(chunks, axis=0)


class PanosetiRun:
    """Represents a PanoSETI observing run (directory)."""
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.products: dict[str, PFFSequence] = {}
        self.configs: dict[str, Any] = {}
        self._load_configs()
        self._scan()

    def _load_configs(self) -> None:
        for f in self.run_dir.glob("*.json"):
            try:
                with open(f, 'rb') as jf:
                    self.configs[f.stem] = orjson.loads(jf.read())
            except Exception:
                pass

    def _scan(self) -> None:
        files_map: dict[str, list[Path]] = {}
        for f in self.run_dir.glob("*.pff"):
            if f.name == 'hk.pff' or f.stat().st_size == 0:
                continue
            parts = f.name.split('.')
            key_parts = [p for p in parts if not (p.startswith('start') or p.startswith('seqno') or p == 'pff')]
            key = ".".join(key_parts)
            if key not in files_map:
                files_map[key] = []
            files_map[key].append(f)

        for k, v in files_map.items():
            try:
                seq = PFFSequence(v)
                if len(seq) > 0:
                    self.products[k] = seq
            except Exception:
                pass

    def list_products(self) -> list[str]:
        return sorted(self.products.keys())

    def get_product(self, product_name: str) -> PFFSequence:
        if product_name not in self.products:
            raise KeyError(f"Product {product_name} not found.")
        return self.products[product_name]

class hkpff:
    """Modernized housekeeping parser."""
    def __init__(self, fn: str = 'hk.pff'):
        self.fn = Path(fn)

    def readhk(self) -> dict[str, dict[str, list[Any]]]:
        hk_info: dict[str, dict[str, list[Any]]] = {}
        if not self.fn.exists():
            return hk_info
        with open(self.fn, 'rb') as f:
            for line in f:
                try:
                    data = orjson.loads(line)
                    for key, values in data.items():
                        if key not in hk_info:
                            hk_info[key] = {k: [] for k in values}
                        for k, v in values.items():
                            target_k = k
                            if k == 'TEMP1': target_k = 'DET_TEMP'
                            if k == 'TEMP2': target_k = 'FPGA_TEMP'
                            
                            if target_k not in hk_info[key]:
                                hk_info[key][target_k] = []
                                
                            try:
                                if '.' in v: hk_info[key][target_k].append(float(v))
                                else: hk_info[key][target_k].append(int(v))
                            except (ValueError, TypeError):
                                hk_info[key][target_k].append(v)
                except Exception:
                    continue
        return hk_info


class qconfig:
    """Modernized config loader."""
    def __init__(self, pattern: str):
        self.config = {}
        base_dir = Path(pattern).parent if '/' in pattern else Path('.')
        glob_pattern = Path(pattern).name
        for f in base_dir.glob(glob_pattern):
            try:
                with open(f, 'rb') as jf:
                    self.config[f.stem] = orjson.loads(jf.read())
            except Exception:
                pass
