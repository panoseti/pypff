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
from .models import QuaboHeader, ModuleHeader, FrameConfig

logger = logging.getLogger(__name__)

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

        self.metadata_offsets: dict[str, tuple[int, int]] = {}

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

    def index_timestamps(self, precise: bool = True) -> None:
        """
        Pre-calculates and caches all frame timestamps.
        Highly recommended before performing multiple seek_time() operations or timeline analysis.
        """
        if self._timestamps is not None:
            return

        logger.info(f"Indexing {self._total_frames:,} timestamps for {self.name}...")
        t0 = time.monotonic()
        
        # Vectorized extraction of all timing components
        keys = ["tv_sec", "tv_usec", "pkt_nsec", "pkt_tai"]
        if "quabo_0.tv_sec" in self.metadata_offsets:
            # Module mode: try all quabos until we find valid time (usually quabo_0)
            # For simplicity and performance, we'll extract quabo_0 components
            # and fallback to others if tv_sec is 0 in post-processing.
            # In practice, researchers usually just need one stable clock.
            data = self.get_metadata_arrays([f"quabo_0.{k}" for k in keys])
            # Remap keys for precise_time helper
            data = {k.split(".")[1]: v for k, v in data.items()}
        else:
            data = self.get_metadata_arrays(keys)

        # Apply timing reconciliation logic in vectorized form
        # Porting get_precise_time_ns logic to NumPy
        tv_sec = data.get("tv_sec", np.zeros(self._total_frames, dtype=np.int64))
        tv_usec = data.get("tv_usec", np.zeros(self._total_frames, dtype=np.int64))
        pkt_nsec = data.get("pkt_nsec", np.zeros(self._total_frames, dtype=np.int64))
        pkt_tai = data.get("pkt_tai", np.zeros(self._total_frames, dtype=np.int64))

        final_sec = tv_sec.copy()
        
        if precise:
            # 1. TAI Reconciliation
            has_tai = pkt_tai != 0
            if np.any(has_tai):
                d = (tv_sec - pkt_tai + 37) % 1024
                final_sec[has_tai & (d == 1)] -= 1
                final_sec[has_tai & (d == 1023)] += 1
            
            # 2. Sub-second Reconciliation (for frames without TAI)
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
        
        # 1. Get vectorized data
        vectorized_data = self.get_metadata_arrays(keys_to_check)
        
        # 2. Compare against naive iteration
        conf = self.frame_config
        for i in range(check_frames):
            file_idx, local_idx = self._locate_frame(i)
            mm = self._get_mmap(file_idx)
            if not mm:
                continue
                
            offset = local_idx * conf.frame_size
            header_bytes = mm[offset : offset + conf.header_size]
            h = orjson.loads(header_bytes)
            
            for key in keys_to_check:
                # Naive value
                if '.' in key:
                    parent, child = key.split('.', 1)
                    naive_val = h.get(parent, {}).get(child, 0)
                else:
                    naive_val = h.get(key, 0)
                    
                # Vectorized value
                vec_val = vectorized_data[key][i]
                
                if int(naive_val) != int(vec_val):
                    print(f"Mismatch at frame {i} for key '{key}': Naive={naive_val}, Vectorized={vec_val}")
                    return False
                    
        print(f"Successfully verified {check_frames} frames against naive JSON parsing.")
        return True

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
            header_bytes = chunk[:self.header_size]
            self._analyze_offsets(header_bytes)

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

    def _analyze_offsets(self, header_bytes: bytes):
        """Dynamically finds byte offsets for numeric fields in the fixed-width JSON header."""
        self.metadata_offsets = {}
        
        # Find positions of "quabo_0", "quabo_1", etc. for Module mode
        q_pos = []
        for i in range(4):
            pos = header_bytes.find(f'"quabo_{i}"'.encode())
            q_pos.append(pos)
            
        is_module = q_pos[0] != -1
        
        # Regex to find "key": <value_string> followed by ,, {, or }
        # This captures the entire padded region (including spaces) so the 
        # byte offset remains constant even if the number of digits changes.
        pattern = re.compile(rb'"(\w+)":([^,{}]+)')
        
        for match in pattern.finditer(header_bytes):
            key = match.group(1).decode()
            val_bytes = match.group(2)
            
            # Check if this looks like a padded integer field
            if not re.match(rb'^\s*-?\d+\s*$', val_bytes):
                continue
                
            val_start = match.start(2)
            val_end = match.end(2)
            
            if is_module:
                # Determine which quabo this belongs to based on previous "quabo_X" tag
                q_idx = -1
                for i in range(3, -1, -1):
                    if q_pos[i] != -1 and q_pos[i] < match.start():
                        q_idx = i
                        break
                if q_idx != -1:
                    full_key = f"quabo_{q_idx}.{key}"
                    self.metadata_offsets[full_key] = (val_start, val_end)
            else:
                self.metadata_offsets[key] = (val_start, val_end)

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

    def get_all_metadata(self) -> dict[str, Any]:
        """
        Retrieves all dynamically mapped integer metadata fields for the entire sequence.
        Returns a dictionary of NumPy arrays. For module mode, keys are nested
        under 'quabo_0', 'quabo_1', etc., mimicking the legacy io.py format.
        """
        flat_data = self.get_metadata_arrays(list(self.metadata_offsets.keys()))
        
        result: dict[str, Any] = {}
        for key, arr in flat_data.items():
            if '.' in key:
                parent, child = key.split('.', 1)
                if parent not in result:
                    result[parent] = {}
                result[parent][child] = arr
            else:
                result[key] = arr
                
        return result

    def get_metadata_arrays(self, keys: list[str]) -> dict[str, np.ndarray]:
        """
        Extracts requested metadata fields for all frames in the sequence using vectorized byte extraction.
        
        Args:
            keys: List of metadata keys (e.g., ['pkt_num', 'tv_sec']).
                 For module mode, use 'quabo_X.key'.
        
        Returns:
            Dictionary mapping keys to NumPy arrays of type int64.
        """
        if not self.frame_config:
            return {}

        results = {k: np.zeros(self._total_frames, dtype=np.int64) for k in keys}
        conf = self.frame_config
        
        frames_processed = 0
        for file_idx, filepath in enumerate(self.file_paths):
            count = self._file_frame_counts[file_idx]
            if count <= 0:
                continue
                
            mm = self._get_mmap(file_idx)
            if not mm:
                frames_processed += count
                continue
                
            for key in keys:
                if key not in self.metadata_offsets:
                    continue
                    
                start, end = self.metadata_offsets[key]
                val_len = end - start
                
                # Create a structured dtype that targets this specific field in every frame
                dt = np.dtype({
                    'names': ['pre', 'val'],
                    'formats': [f'V{start}', f'S{val_len}'],
                    'itemsize': conf.frame_size
                })
                
                # Read all values for this file instantly
                raw_strings = np.frombuffer(mm, dtype=dt, count=count)['val']
                
                # Convert numeric strings to int64 in bulk
                results[key][frames_processed : frames_processed + count] = raw_strings.astype(np.int64)
                
            frames_processed += count
            
        return results

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
        start_offset = conf.header_size + 1

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

                # Vectorized extraction of unaligned payloads using structured dtypes
                dt = np.dtype({
                    'names': ['header', 'payload'],
                    'formats': [f'V{start_offset}', f'V{conf.payload_size}'],
                    'itemsize': conf.frame_size
                })
                
                byte_offset = local_start * conf.frame_size
                raw_payloads = np.frombuffer(mm, dtype=dt, count=to_read, offset=byte_offset)['payload']
                
                # Convert void items to target array via buffer copy
                chunk = np.frombuffer(raw_payloads.tobytes(), dtype=dtype).reshape(to_read, *conf.image_shape)
                chunks.append(chunk)

            frames_collected += to_read
            current_global += to_read

        if len(chunks) == 1:
            # Sliced views from tobtyes() are already copies, but if we used a direct view (rare) we'd copy
            return chunks[0]

        return np.concatenate(chunks, axis=0)

    def get_frame_time(self, idx: int, precise: bool = True) -> int:
        """
        Retrieves ONLY the nanosecond timestamp for a frame.
        Fastest way to get time.
        """
        if precise and self._timestamps is not None:
            return int(self._timestamps[idx])

        file_idx, local_idx = self._locate_frame(idx)
        conf = self.frame_config
        if not conf: return 0
        mm = self._get_mmap(file_idx)
        if not mm: return 0
        
        offset = local_idx * conf.frame_size

        # Fast path: use dynamically discovered byte offsets
        if self.metadata_offsets:
            is_module = "quabo_0.tv_sec" in self.metadata_offsets
            prefix = "quabo_0." if is_module else ""
            
            def get_val(k):
                start, end = self.metadata_offsets[prefix + k]
                return int(mm[offset + start : offset + end])

            tv_sec = get_val("tv_sec")
            tv_usec = get_val("tv_usec")
            pkt_nsec = get_val("pkt_nsec")
            pkt_tai = get_val("pkt_tai")
            
            if precise:
                return get_precise_time_ns(tv_sec, tv_usec, pkt_nsec, pkt_tai)
            else:
                return get_coarse_time_ns(tv_sec, tv_usec)

        # Fallback to JSON parsing
        header_bytes = mm[offset: offset + conf.header_size]
        h = orjson.loads(header_bytes)

        if 'quabo_0' in h:
            for i in range(4):
                q = h[f'quabo_{i}']
                if q['tv_sec'] != 0:
                    break
            else:
                q = h['quabo_0']
            
            if precise:
                return get_precise_time_ns(q['tv_sec'], q['tv_usec'], q['pkt_nsec'], q.get('pkt_tai', 0))
            else:
                return get_coarse_time_ns(q['tv_sec'], q['tv_usec'])
        elif 'pkt_nsec' in h:
            if precise:
                return get_precise_time_ns(h['tv_sec'], h['tv_usec'], h['pkt_nsec'], h.get('pkt_tai', 0))
            else:
                return get_coarse_time_ns(h['tv_sec'], h['tv_usec'])
        else:
            return 0

    def seek_time(self, target_time_ns: int) -> int:
        """
        Binary Search for frame index closest to target_time_ns.
        """
        if self._total_frames == 0:
            return 0

        low = 0
        high = self._total_frames - 1

        t_start = self.get_frame_time(low)
        if target_time_ns <= t_start: return low

        t_end = self.get_frame_time(high)
        if target_time_ns >= t_end: return high

        while low <= high:
            mid = (low + high) // 2
            t_mid = self.get_frame_time(mid)

            if t_mid < target_time_ns:
                low = mid + 1
            elif t_mid > target_time_ns:
                high = mid - 1
            else:
                return mid

        candidates = [c for c in [high, low] if 0 <= c < self._total_frames]
        return min(candidates, key=lambda i: abs(self.get_frame_time(i) - target_time_ns))

    def to_dask(self) -> Any:
        """Convert to a Dask Array for distributed processing."""
        try:
            import dask.array as da
        except ImportError:
            raise ImportError("Dask is not installed. Install with 'uv add dask' or use the [dev] extra.")

        if not self.frame_config:
            raise RuntimeError("Frame configuration not analyzed.")

        return None


class PanosetiRun:
    """Represents a PanoSETI observing run (directory)."""
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.products: dict[str, PFFSequence] = {}
        self.configs: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self._load_configs()
        self._scan()

    def _load_configs(self) -> None:
        from .models import DataConfig, ObsConfig, DaqConfig, QuaboConfig, PhBaselineConfig
        
        # 1. JSON files
        for f in self.run_dir.glob("*.json"):
            try:
                with open(f, 'rb') as jf:
                    data = orjson.loads(jf.read())
                    
                    # Attempt to validate with models based on filename
                    stem = f.stem
                    if stem == "data_config":
                        self.configs[stem] = DataConfig(**data)
                    elif stem == "obs_config":
                        self.configs[stem] = ObsConfig(**data)
                    elif stem == "daq_config":
                        self.configs[stem] = DaqConfig(**data)
                    elif stem.startswith("quabo_config"):
                        self.configs[stem] = QuaboConfig(**data)
                    elif stem == "quabo_ph_baseline":
                        self.configs[stem] = PhBaselineConfig(**data)
                    else:
                        self.configs[stem] = data
            except Exception:
                # Fallback to raw dict if validation fails
                try:
                    with open(f, 'rb') as jf:
                        self.configs[f.stem] = orjson.loads(jf.read())
                except Exception:
                    pass

        # 2. TOML files
        for f in self.run_dir.glob("*.toml"):
            try:
                with open(f, "rb") as tf:
                    self.configs[f.stem] = tomllib.load(tf)
            except Exception:
                pass

        # 3. JSONL files (like hp_stdout.jsonl)
        for f in self.run_dir.glob("*.jsonl"):
            try:
                with open(f, "rb") as jlf:
                    lines = jlf.readlines()
                    self.metadata[f.name] = [orjson.loads(line) for line in lines if line.strip()]
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

    def list_logs(self) -> list[str]:
        """List all log files in the run directory."""
        logs = []
        for ext in ["*.log", "*.txt"]:
            logs.extend([f.name for f in self.run_dir.glob(ext)])
        return sorted(logs)

    def get_log(self, name: str) -> str:
        """Read and return the content of a log file."""
        log_path = self.run_dir / name
        if not log_path.exists():
            raise FileNotFoundError(f"Log file {name} not found.")
        return log_path.read_text()

    def get_manifest(self) -> dict[str, dict[str, Any]]:
        """Parse the manifest file and return a dictionary of entries."""
        manifest_files = list(self.run_dir.glob("dp_manifest.*.txt"))
        if not manifest_files:
            return {}
        
        # Take the first one (usually only one per node)
        m_file = manifest_files[0]
        entries = {}
        with open(m_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("  ", 3)
                if len(parts) == 4:
                    digest, size, mtime, relpath = parts
                    entries[relpath] = {
                        "digest": digest,
                        "size": int(size),
                        "mtime": int(mtime)
                    }
        return entries

    def show(self, details: bool = False) -> None:
        """Rich visualization of the Run structure.
        
        Args:
            details: If True, list individual PFF files within each product.
        """
        console = Console()

        run_name = self.run_dir.resolve().name
        tree = Tree(f"[bold gold1]Run: {run_name}[/]")
        
        # 1. Configs & Metadata
        if self.configs or self.metadata:
            config_branch = tree.add("Configurations & Metadata")
            for k in sorted(self.configs.keys()):
                # Distinguish by extension
                if (self.run_dir / f"{k}.json").exists():
                    config_branch.add(f"[cyan]{k}.json[/]")
                elif (self.run_dir / f"{k}.toml").exists():
                    config_branch.add(f"[yellow]{k}.toml[/]")
                else:
                    config_branch.add(f"[cyan]{k}[/]")
            
            for k in sorted(self.metadata.keys()):
                config_branch.add(f"[blue]{k}[/]")

        # 2. Logs
        logs = self.list_logs()
        if logs:
            log_branch = tree.add("Logs")
            for l in logs:
                log_branch.add(f"[magenta]{l}[/]")

        # 3. Manifests
        manifests = list(self.run_dir.glob("dp_manifest.*.txt"))
        if manifests:
            manifest_branch = tree.add("Data Manifests")
            for m in manifests:
                manifest_branch.add(f"[green]{m.name}[/]")

        # 4. Products
        if self.products:
            prod_branch = tree.add("Data Products")
            for name, seq in sorted(self.products.items()):
                info = f"[bold green]{name}[/] ({len(seq):,} frames)"
                b = prod_branch.add(info)
                if details:
                    for f in seq.file_paths:
                        b.add(f"[dim]{f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)[/]")
        
        # 5. Housekeeping
        if (self.run_dir / "hk.pff").exists():
            hk_size = (self.run_dir / "hk.pff").stat().st_size
            tree.add(f"[bold white]Housekeeping:[/] hk.pff ({hk_size} bytes)")

        console.print(tree)

class hkpff:
    """Modernized housekeeping parser."""
    def __init__(self, fn: str = 'hk.pff'):
        self.fn = Path(fn)

    def readhk(self) -> dict[str, dict[str, list[Any]]]:
        hk_info: dict[str, dict[str, list[Any]]] = {}
        if not self.fn.exists():
            return hk_info
            
        with open(self.fn, 'rb') as f:
            # PFF JSON records are delimited by \n\n
            content = f.read()
            records = content.split(b'\n\n')
            
            for rec in records:
                if not rec.strip():
                    continue
                try:
                    data = orjson.loads(rec)
                    for key, values in data.items():
                        if key not in hk_info:
                            hk_info[key] = {}
                        
                        for k, v in values.items():
                            target_k = k
                            if k == 'TEMP1': target_k = 'DET_TEMP'
                            if k == 'TEMP2': target_k = 'FPGA_TEMP'
                            
                            if target_k not in hk_info[key]:
                                hk_info[key][target_k] = []
                                
                            try:
                                if isinstance(v, str) and '.' in v:
                                    hk_info[key][target_k].append(float(v))
                                elif isinstance(v, str) and v.isdigit():
                                    hk_info[key][target_k].append(int(v))
                                else:
                                    hk_info[key][target_k].append(v)
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
