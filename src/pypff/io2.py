import bisect
import logging
import mmap
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import orjson
from pydantic import ValidationError
from rich.console import Console
from rich.tree import Tree

from .utils import (
    parse_filename,
    extract_seqno,
    get_coarse_time_ns,
    get_precise_time_ns,
)
from .models import FrameConfig, ModuleHeader, QuaboHeader

logger = logging.getLogger(__name__)

# --- Core I/O Classes ---

class PFFSequence:
    """
    Represents a time-ordered sequence of PFF files for a single data product.
    Zero-copy access via mmap. Supports highly optimized Sequential and Random Access.
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

        self.metadata_offsets: dict[str, tuple[int, int]] = {}

        self._analyze_structure()
        self._index_files()

    def __getstate__(self) -> dict[str, Any]:
        """Prepare for pickling: remove non-pickleable file handles and mmaps."""
        state = self.__dict__.copy()
        state['_open_mmaps'] = {}
        state['_open_files'] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore after unpickling."""
        self.__dict__.update(state)
        # Handles will be lazily re-opened on first access

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

    def index_timestamps(self, precise: bool = True) -> None:
        if self._timestamps is not None:
            return

        logger.info(f"Indexing {self._total_frames:,} timestamps for {self.name}...")
        t0 = time.monotonic()
        
        keys = ["tv_sec", "tv_usec", "pkt_nsec", "pkt_tai"]
        if "quabo_0.tv_sec" in self.metadata_offsets:
            data = self.get_metadata_arrays([f"quabo_0.{k}" for k in keys])
            data = {k.split(".")[1]: v for k, v in data.items()}
        else:
            data = self.get_metadata_arrays(keys)

        tv_sec = data.get("tv_sec", np.zeros(self._total_frames, dtype=np.int64))
        tv_usec = data.get("tv_usec", np.zeros(self._total_frames, dtype=np.int64))
        pkt_nsec = data.get("pkt_nsec", np.zeros(self._total_frames, dtype=np.int64))
        pkt_tai = data.get("pkt_tai", np.zeros(self._total_frames, dtype=np.int64))

        final_sec = tv_sec.copy()
        
        if precise:
            # Check for non-zero pkt_tai to determine if we use the precise PanoSETI correction
            has_tai = pkt_tai != 0
            if np.any(has_tai):
                d = (tv_sec - pkt_tai + 37) % 1024
                final_sec[has_tai & (d == 1)] -= 1
                final_sec[has_tai & (d == 1023)] += 1
            
            no_tai = ~has_tai
            if np.any(no_tai):
                tv_nsec_equiv = tv_usec[no_tai] * 1000
                diff = tv_nsec_equiv - pkt_nsec[no_tai]
                
                final_sec_no_tai = final_sec[no_tai]
                final_sec_no_tai[diff > 500_000_000] += 1
                final_sec_no_tai[diff < -500_000_000] -= 1
                final_sec[no_tai] = final_sec_no_tai
            
            self._timestamps = (final_sec * 1_000_000_000) + pkt_nsec
        else:
            self._timestamps = (tv_sec * 1_000_000_000) + (tv_usec * 1_000)

        elapsed = time.monotonic() - t0
        logger.info(f"Indexed {self._total_frames:,} frames in {elapsed:.2f}s")

    def get_frame_time(self, idx: int, precise: bool = True) -> int:
        """Returns the nanosecond timestamp for a single frame."""
        if self._timestamps is not None:
            return int(self._timestamps[idx])
        
        # Single frame metadata fetch for timing
        keys = ["tv_sec", "tv_usec", "pkt_nsec", "pkt_tai"]
        if "quabo_0.tv_sec" in self.metadata_offsets:
            m_keys = [f"quabo_0.{k}" for k in keys]
            data = self.get_metadata_arrays(m_keys, indices=[idx])
            data = {k.split(".")[1]: v[0] for k, v in data.items()}
        else:
            data = self.get_metadata_arrays(keys, indices=[idx])
            data = {k: v[0] for k, v in data.items()}
        
        if precise:
            return get_precise_time_ns(data['tv_sec'], data['tv_usec'], data['pkt_nsec'], data['pkt_tai'])
        return (data['tv_sec'] * 1_000_000_000) + (data['tv_usec'] * 1_000)

    def seek_time(self, timestamp_ns: int) -> int:
        """Finds the index of the frame closest to the given timestamp using binary search."""
        self.index_timestamps()
        if self._timestamps is None: return 0
        idx = bisect.bisect_left(self._timestamps, timestamp_ns)
        if idx == 0: return 0
        if idx == len(self._timestamps): return idx - 1
        # Return the closer of the two
        if timestamp_ns - self._timestamps[idx - 1] < self._timestamps[idx] - timestamp_ns:
            return idx - 1
        return idx

    def get_all_metadata(self) -> dict[str, Any]:
        """Returns all dynamically discovered metadata fields for the entire sequence in a structured format."""
        flat = self.get_metadata_arrays(list(self.metadata_offsets.keys()))
        structured: dict[str, Any] = {}
        for k, v in flat.items():
            if '.' in k:
                parent, child = k.split('.', 1)
                if parent not in structured: structured[parent] = {}
                structured[parent][child] = v
            else:
                structured[k] = v
        return structured

    def print_metadata_offsets(self) -> None:
        """Prints the dynamically determined byte offsets for all numeric metadata fields."""
        print(f"Metadata Byte Offsets for {self.name}:")
        if not self.metadata_offsets:
            print("  (No offsets determined)")
            return
        for key, (start, end) in sorted(self.metadata_offsets.items()):
            print(f"  {key}: bytes {start} to {end} (length: {end - start})")

    def verify_metadata_offsets(self, num_frames: int = 100) -> bool:
        """
        Sanity check: compares the vectorized metadata extraction against naive JSON parsing
        for a subset of frames to ensure the byte offsets are perfectly aligned.
        
        Args:
            num_frames: Number of frames to check. Defaults to 100.
            
        Returns:
            True if all checked frames match, False otherwise.
        """
        if self._total_frames == 0 or not self.frame_config or not self.metadata_offsets:
            return True
        check_frames = min(num_frames, self._total_frames)
        keys_to_check = list(self.metadata_offsets.keys())
        vectorized_data = self.get_metadata_arrays(keys_to_check)
        
        conf = self.frame_config
        for i in range(check_frames):
            file_idx, local_idx = self._locate_frame(i)
            mm = self._get_mmap(file_idx)
            if not mm: continue
                
            offset = local_idx * conf.frame_size
            header_bytes = mm[offset : offset + conf.header_size]
            h = orjson.loads(header_bytes)
            
            for key in keys_to_check:
                if '.' in key:
                    parent, child = key.split('.', 1)
                    naive_val = h.get(parent, {}).get(child, 0)
                else:
                    naive_val = h.get(key, 0)
                vec_val = vectorized_data[key][i]
                if int(naive_val) != int(vec_val):
                    print(f"Mismatch at frame {i} for key '{key}': Naive={naive_val}, Vectorized={vec_val}")
                    return False
        return True

    def _analyze_structure(self) -> None:
        """Determines frame structure and enforces exact PanoSETI shapes and types."""
        sample_file = next((p for p in self.file_paths if p.stat().st_size > 0), None)
        if not sample_file: return

        with open(sample_file, 'rb') as f:
            chunk = f.read(4096)
            match = re.search(b'}\n\n\\*', chunk) or re.search(b'\n\n\\*', chunk)
            if not match: raise ValueError(f"Invalid PFF format in {sample_file}")

            self.header_size = match.end() - 1
            self._analyze_offsets(chunk[:self.header_size])

            fmt = str(self.meta.get('dp', 'unknown')).lower()
            if 'img8' in fmt: shape, dtype, bpp = (32, 32), np.uint8, 1
            elif 'img16' in fmt: shape, dtype, bpp = (32, 32), np.uint16, 2
            elif 'ph256' in fmt: shape, dtype, bpp = (16, 16), np.int16, 2
            else: shape, dtype, bpp = (32, 32), np.int16, 2

            payload_size = shape[0] * shape[1] * bpp
            self.frame_config = FrameConfig(
                header_size=self.header_size,
                payload_size=payload_size,
                frame_size=self.header_size + 1 + payload_size,
                image_shape=shape,
                dtype_str=np.dtype(dtype).name,
                bytes_per_pixel=bpp,
                format_name=fmt
            )

    def _analyze_offsets(self, header_bytes: bytes):
        """Dynamically finds byte offsets for numeric fields in the fixed-width JSON header."""
        self.metadata_offsets = {}
        q_pos = [header_bytes.find(f'"quabo_{i}"'.encode()) for i in range(4)]
        is_module = q_pos[0] != -1
        
        pattern = re.compile(rb'"(\w+)":([^,{}]+)')
        for match in pattern.finditer(header_bytes):
            key, val_bytes = match.group(1).decode(), match.group(2)
            if not re.match(rb'^\s*-?\d+\s*$', val_bytes): continue
            start, end = match.start(2), match.end(2)
            
            if is_module:
                q_idx = next((i for i in range(3, -1, -1) if q_pos[i] != -1 and q_pos[i] < match.start()), -1)
                if q_idx != -1: self.metadata_offsets[f"quabo_{q_idx}.{key}"] = (start, end)
            else:
                self.metadata_offsets[key] = (start, end)

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

    def __getitem__(self, key: int | slice) -> np.ndarray:
        """
        Syntactic sugar for get_image_array. 
        Supports single index (seq[0]) or slicing (seq[0:100:10]).
        Returns a NumPy array.
        """
        if isinstance(key, int):
            if key < 0: key += self._total_frames
            if not (0 <= key < self._total_frames): raise IndexError("Index out of range")
            return self.get_image_array(indices=[key])[0]
        elif isinstance(key, slice):
            return self.get_image_array(slice_obj=key)
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    def _get_mmap(self, file_idx: int) -> Optional[mmap.mmap]:
        if file_idx in self._open_mmaps: return self._open_mmaps[file_idx]
        try:
            f = open(self.file_paths[file_idx], 'rb')
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            self._open_files[file_idx] = f
            self._open_mmaps[file_idx] = mm
            return mm
        except ValueError:
            return None

    def _locate_frame(self, idx: int) -> tuple[int, int]:
        if not (0 <= idx < self._total_frames): raise IndexError(f"Frame {idx} out of range")
        file_idx = bisect.bisect_right(self._cumulative_frames, idx)
        prev_limit = self._cumulative_frames[file_idx - 1] if file_idx > 0 else 0
        return file_idx, idx - prev_limit

    def get_frame(self, idx: int, raw: bool = False) -> tuple[QuaboHeader | ModuleHeader | dict[str, Any], np.ndarray]:
        if not self.frame_config: raise RuntimeError("Frame config missing.")
        file_idx, local_idx = self._locate_frame(idx)
        conf = self.frame_config
        mm = self._get_mmap(file_idx)
        if not mm: raise RuntimeError(f"No mmap for file {file_idx}")

        offset = local_idx * conf.frame_size
        header_end = offset + conf.header_size
        h_dict = orjson.loads(mm[offset:header_end])

        header_obj = h_dict if raw else (
            ModuleHeader(**h_dict) if 'quabo_0' in h_dict else 
            QuaboHeader(**h_dict) if 'quabo_num' in h_dict else h_dict
        )

        img_start = header_end + 1
        img = np.frombuffer(mm[img_start:img_start + conf.payload_size], dtype=conf.dtype).reshape(conf.image_shape)
        return header_obj, img

    def get_metadata_arrays(self, keys: list[str], indices: Optional[Sequence[int] | np.ndarray] = None, slice_obj: Optional[slice] = None) -> dict[str, np.ndarray]:
        """
        Extracts specific metadata fields. 
        Supports Sequential (entire sequence), Sliced (slice_obj), or Random (indices) access.
        """
        if not self.frame_config: return {}
        conf = self.frame_config

        if slice_obj is not None:
            indices = np.arange(*slice_obj.indices(self._total_frames))
        
        # --- RANDOM ACCESS PATH ---
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            count = len(indices)
            results = {k: np.zeros(count, dtype=np.int64) for k in keys}
            if count == 0: return results
            
            # Sort for sequential disk reading to avoid thrashing
            sort_order = np.argsort(indices)
            sorted_idx = indices[sort_order]
            
            for res_idx, global_idx in enumerate(sorted_idx):
                if not (0 <= global_idx < self._total_frames): continue
                f_idx, l_idx = self._locate_frame(global_idx)
                mm = self._get_mmap(f_idx)
                if not mm: continue
                
                offset = l_idx * conf.frame_size
                for k in keys:
                    if k in self.metadata_offsets:
                        start, end = self.metadata_offsets[k]
                        results[k][sort_order[res_idx]] = int(mm[offset + start : offset + end])
            return results

        # --- SEQUENTIAL / BULK ACCESS PATH ---
        results = {k: np.zeros(self._total_frames, dtype=np.int64) for k in keys}
        frames_processed = 0
        for file_idx, count in enumerate(self._file_frame_counts):
            if count <= 0: continue
            mm = self._get_mmap(file_idx)
            if not mm:
                frames_processed += count
                continue
                
            for key in keys:
                if key not in self.metadata_offsets: continue
                start, end = self.metadata_offsets[key]
                dt = np.dtype({'names': ['pre', 'val'], 'formats': [f'V{start}', f'S{end - start}'], 'itemsize': conf.frame_size})
                results[key][frames_processed : frames_processed + count] = np.frombuffer(mm, dtype=dt, count=count)['val'].astype(np.int64)
            frames_processed += count
        return results

    def get_image_array(self, start: int = 0, count: Optional[int] = None, indices: Optional[Sequence[int] | np.ndarray] = None, slice_obj: Optional[slice] = None) -> np.ndarray:
        """
        Retrieves image frames.
        Supports Sequential (start, count), Random (indices), or Sliced (slice_obj) access.
        Uses zero-copy views where possible.
        """
        if not self.frame_config: return np.empty(0)
        conf = self.frame_config
        start_offset = conf.header_size + 1
        base_strides = np.empty(conf.image_shape, dtype=conf.dtype).strides

        if slice_obj is not None:
            indices = np.arange(*slice_obj.indices(self._total_frames))

        # --- RANDOM ACCESS PATH (Includes Slicing) ---
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            n_frames = len(indices)
            if n_frames == 0: return np.empty((0, *conf.image_shape), dtype=conf.dtype)
            
            result = np.empty((n_frames, *conf.image_shape), dtype=conf.dtype)
            
            # Optimization: If indices are mostly sequential within files, this is efficient.
            # We don't sort here to preserve the requested order (e.g. for negative steps).
            for res_idx, global_idx in enumerate(indices):
                if not (0 <= global_idx < self._total_frames): continue
                f_idx, l_idx = self._locate_frame(global_idx)
                mm = self._get_mmap(f_idx)
                if not mm: continue
                
                offset = (l_idx * conf.frame_size) + start_offset
                result[res_idx] = np.ndarray(
                    shape=conf.image_shape, dtype=conf.dtype, buffer=mm, offset=offset, strides=base_strides
                )
            return result

        # --- SEQUENTIAL / BULK ACCESS PATH ---
        count = min(count or (self._total_frames - start), self._total_frames - start)
        if count <= 0: return np.empty((0, *conf.image_shape), dtype=conf.dtype)

        result = np.empty((count, *conf.image_shape), dtype=conf.dtype)
        mmap_strides = (conf.frame_size,) + base_strides

        frames_collected, current_global = 0, start
        while frames_collected < count:
            file_idx, local_start = self._locate_frame(current_global)
            to_read = min(count - frames_collected, self._file_frame_counts[file_idx] - local_start)

            if to_read > 0:
                mm = self._get_mmap(file_idx)
                if mm:
                    chunk_view = np.ndarray(
                        shape=(to_read, *conf.image_shape), dtype=conf.dtype, buffer=mm, 
                        offset=(local_start * conf.frame_size) + start_offset, strides=mmap_strides
                    )
                    result[frames_collected : frames_collected + to_read] = chunk_view

            frames_collected += to_read
            current_global += to_read

        return result


class PanosetiRun:
    """
    Unified entrypoint for a PanoSETI observing run.
    Provides lazy-loaded access to configs, metadata logs, housekeeping telemetry, and data products.
    """
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.products: dict[str, PFFSequence] = {}
        
        # Internal cache for lazy loading
        self._configs: Optional[dict[str, Any]] = None
        self._metadata: Optional[dict[str, list[dict]]] = None
        self._hk_data: Optional[dict[str, dict[str, list[Any]]]] = None
        self._manifests: Optional[dict[str, str]] = None
        
        self._scan_products()

    def _scan_products(self) -> None:
        files_map: dict[str, list[Path]] = {}
        for f in self.run_dir.glob("*.pff"):
            if f.name == 'hk.pff' or f.stat().st_size == 0: continue
            key = ".".join([p for p in f.name.split('.') if not (p.startswith('start') or p.startswith('seqno') or p == 'pff')])
            if key not in files_map: files_map[key] = []
            files_map[key].append(f)

        for k, v in files_map.items():
            try:
                seq = PFFSequence(v)
                if len(seq) > 0: self.products[k] = seq
            except Exception: pass

    # --- Data Products ---
    def list_products(self) -> list[str]:
        return sorted(self.products.keys())

    def get_product(self, product_name: str) -> PFFSequence:
        if product_name not in self.products: raise KeyError(f"Product {product_name} not found.")
        return self.products[product_name]

    # --- Lazy Loaded Properties (Compatible with existing notebooks) ---
    @property
    def configs(self) -> dict[str, Any]:
        if self._configs is None: self._load_configs()
        return self._configs

    @property
    def metadata(self) -> dict[str, list[dict]]:
        if self._metadata is None: self._load_metadata()
        return self._metadata

    @property
    def manifests(self) -> dict[str, str]:
        if self._manifests is None: self._load_manifests()
        return self._manifests

    # --- Configurations ---
    def _load_configs(self) -> None:
        from .models import (
            DataConfig, ObsConfig, DaqConfig, QuaboConfig, QuaboPhBaseline,
            PhBaselineConfig, NetworkConfig, QuaboUids, FirmwareConfig
        )
        
        self._configs = {}
        models_map = {
            "data_config": DataConfig, 
            "obs_config": ObsConfig, 
            "daq_config": DaqConfig, 
            "quabo_ph_baseline": QuaboPhBaseline,
            "ph_baseline_config": PhBaselineConfig,
            "network_config": NetworkConfig,
            "quabo_uids": QuaboUids,
            "firmware_config": FirmwareConfig
        }

        for f in self.run_dir.glob("*.json"):
            if f.name.startswith("dp_manifest"): continue
            try:
                data = orjson.loads(f.read_bytes())
                model = models_map.get(f.stem) or (QuaboConfig if f.stem.startswith("quabo_config") else None)
                self._configs[f.stem] = model(**data) if model else data
            except Exception: pass

        for f in self.run_dir.glob("*.toml"):
            try:
                import tomllib
                self._configs[f.stem] = tomllib.loads(f.read_text())
            except Exception: pass

    def get_config(self, name: str) -> Any:
        clean_name = name.replace('.json', '').replace('.toml', '')
        return self.configs.get(clean_name)

    # --- JSONL Metadata ---
    def _load_metadata(self) -> None:
        self._metadata = {}
        for f in self.run_dir.glob("*.jsonl"):
            try:
                self._metadata[f.name] = [orjson.loads(line) for line in f.read_text().splitlines() if line.strip()]
            except Exception: pass

    def get_metadata_log(self, name: str) -> list[dict]:
        return self.metadata.get(name, [])

    # --- Manifests ---
    def _load_manifests(self) -> None:
        self._manifests = {}
        for f in self.run_dir.glob("dp_manifest*"):
            try:
                self._manifests[f.name] = f.read_text()
            except Exception: pass

    def get_manifest(self, name: str) -> str:
        return self.manifests.get(name, "")

    # --- Standard Logs ---
    def list_logs(self) -> list[str]:
        return sorted(f.name for ext in ["*.log", "*.txt"] for f in self.run_dir.glob(ext) if not f.name.startswith("dp_manifest"))

    def get_log(self, name: str) -> str:
        log_path = self.run_dir / name
        if not log_path.exists(): raise FileNotFoundError(f"Log {name} not found.")
        return log_path.read_text()

    # --- Housekeeping ---
    def get_hk(self) -> dict[str, dict[str, list[Any]]]:
        """Lazy-loads and parses hk.pff efficiently without blowing up memory."""
        if self._hk_data is not None: return self._hk_data

        hk_file, hk_info = self.run_dir / 'hk.pff', {}
        if not hk_file.exists() or hk_file.stat().st_size == 0:
            self._hk_data = hk_info; return hk_info
            
        key_map = {'TEMP1': 'DET_TEMP', 'TEMP2': 'FPGA_TEMP'}
        with open(hk_file, 'rb') as f:
            try: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError: return hk_info
                
            start, size = 0, mm.size()
            while start < size:
                end = mm.find(b'\n\n', start)
                chunk, start = (mm[start:], size) if end == -1 else (mm[start:end], end + 2)
                if not chunk.strip(): continue
                    
                try:
                    for top_key, values in orjson.loads(chunk).items():
                        target_dict = hk_info.setdefault(top_key, {})
                        for k, v in values.items():
                            target_k = key_map.get(k, k)
                            if target_k not in target_dict: target_dict[target_k] = []
                            if isinstance(v, str):
                                try: v = float(v) if '.' in v else int(v)
                                except ValueError: pass
                            target_dict[target_k].append(v)
                except Exception: continue
            mm.close()
            
        self._hk_data = hk_info
        return hk_info

    # --- Display ---
    def show(self, details: bool = False) -> None:
        console = Console()
        tree = Tree(f"[bold gold1]Run: {self.run_dir.resolve().name}[/]")
        
        # Trigger lazy loads to populate tree correctly
        _ = self.configs; _ = self.metadata; _ = self.manifests
        
        if self._configs or self._metadata:
            cb = tree.add("Configurations & Metadata")
            for k in sorted(self._configs.keys()): cb.add(f"[cyan]{k}[/]")
            for k in sorted(self._metadata.keys()): cb.add(f"[blue]{k}[/]")

        if self._manifests:
            mb = tree.add("Data Manifests")
            for k in sorted(self._manifests.keys()): mb.add(f"[green]{k}[/]")

        logs = self.list_logs()
        if logs:
            lb = tree.add("Logs")
            for l in logs: lb.add(f"[magenta]{l}[/]")
        if self.products:
            pb = tree.add("Data Products")
            for name, seq in sorted(self.products.items()):
                b = pb.add(f"[bold green]{name}[/] ({len(seq):,} frames)")
                if details:
                    for f in seq.file_paths: b.add(f"[dim]{f.name} ({f.stat().st_size / 1024**2:.1f} MB)[/]")
        
        if (self.run_dir / "hk.pff").exists():
            tree.add(f"[bold white]Housekeeping:[/] hk.pff ({(self.run_dir / 'hk.pff').stat().st_size / 1024:.1f} KB)")

        console.print(tree)


# --- Legacy Compatibility Shims ---

class hkpff:
    """Legacy compatibility shim for reading hk.pff files."""
    def __init__(self, fn: str | Path = "hk.pff"):
        self.fn = Path(fn)
        self.hk_info: dict[str, dict[str, list[Any]]] = {}

    def readhk(self) -> dict[str, dict[str, list[Any]]]:
        if not self.fn.exists() or self.fn.stat().st_size == 0:
            return self.hk_info
            
        key_map = {'TEMP1': 'DET_TEMP', 'TEMP2': 'FPGA_TEMP'}
        with open(self.fn, 'rb') as f:
            try: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except ValueError: return self.hk_info
                
            start, size = 0, mm.size()
            while start < size:
                end = mm.find(b'\n\n', start)
                chunk, start = (mm[start:], size) if end == -1 else (mm[start:end], end + 2)
                if not chunk.strip(): continue
                try:
                    for top_key, values in orjson.loads(chunk).items():
                        target_dict = self.hk_info.setdefault(top_key, {})
                        for k, v in values.items():
                            target_k = key_map.get(k, k)
                            if target_k not in target_dict: target_dict[target_k] = []
                            if isinstance(v, str):
                                try: v = float(v) if '.' in v else int(v)
                                except ValueError: pass
                            target_dict[target_k].append(v)
                except Exception: continue
            mm.close()
        return self.hk_info


class qconfig:
    """Legacy compatibility shim for loading configuration JSON files."""
    def __init__(self, fn: str):
        from glob import glob
        self.config: dict[str, Any] = {}
        jfiles = glob(fn)
        if not jfiles:
            raise Exception(f"The config file({fn}) can not be found!")
        
        from .models import QuaboConfig
        for file in jfiles:
            p = Path(file)
            key = p.stem
            try:
                data = orjson.loads(p.read_bytes())
                if key.startswith('quabo_config'):
                    # Use QuaboConfig model to handle the complex parsing, then convert back to dict
                    self.config[key] = QuaboConfig(**data).model_dump()
                else:
                    self.config[key] = data
            except Exception:
                continue