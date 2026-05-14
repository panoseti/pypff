from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from .io2 import PanosetiRun, PFFSequence

logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    test_name: str
    product_name: str
    total_frames: int
    elapsed_seconds: float
    fps: float
    throughput_mbs: float
    extra_meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "product_name": self.product_name,
            "total_frames": self.total_frames,
            "elapsed_seconds": self.elapsed_seconds,
            "fps": self.fps,
            "throughput_mbs": self.throughput_mbs,
            **self.extra_meta
        }

class Profiler:
    """
    Performance profiling utility for PanoSETI data I/O.
    Strictly uses io2.py for benchmarking.
    """

    def __init__(self, run_dir: str | Path):
        self.run = PanosetiRun(run_dir)
        self.results: list[ProfileResult] = []

    def profile_sequential_read(self, product_name: str, n_frames: int | None = None) -> ProfileResult:
        seq = self.run.get_product(product_name)
        n_frames = min(n_frames or len(seq), len(seq))
        
        t0 = time.perf_counter()
        # Read frames one by one
        for i in range(n_frames):
            _ = seq.get_frame(i)
        elapsed = time.perf_counter() - t0
        
        return self._record_result("Sequential Read", product_name, n_frames, elapsed, seq)

    def profile_bulk_read(self, product_name: str, n_frames: int | None = None) -> ProfileResult:
        seq = self.run.get_product(product_name)
        n_frames = min(n_frames or len(seq), len(seq))
        
        t0 = time.perf_counter()
        # Single bulk read call
        _ = seq.read_images_range(start=0, count=n_frames)
        elapsed = time.perf_counter() - t0
        
        return self._record_result("Bulk Read (read_images_range)", product_name, n_frames, elapsed, seq)

    def profile_strided_read(self, product_name: str, step: int = 10, n_frames: int | None = None) -> ProfileResult:
        seq = self.run.get_product(product_name)
        total_available = len(seq)
        indices = np.arange(0, total_available, step)
        if n_frames:
            indices = indices[:n_frames]
        
        n_collected = len(indices)
        
        t0 = time.perf_counter()
        # Slicing syntax
        _ = seq[::step] if not n_frames else seq[0:indices[-1]+1:step]
        elapsed = time.perf_counter() - t0
        
        return self._record_result(f"Strided Read (step={step})", product_name, n_collected, elapsed, seq)

    def profile_random_access(self, product_name: str, n_samples: int = 100) -> ProfileResult:
        seq = self.run.get_product(product_name)
        if len(seq) == 0:
            raise ValueError("Sequence is empty")
            
        indices = np.random.randint(0, len(seq), size=n_samples)
        
        t0 = time.perf_counter()
        _ = seq.read_images(indices=indices)
        elapsed = time.perf_counter() - t0
        
        return self._record_result("Random Access", product_name, n_samples, elapsed, seq)

    def profile_metadata_extraction(self, product_name: str, keys: list[str] | None = None) -> ProfileResult:
        seq = self.run.get_product(product_name)
        if not keys:
            keys = ["tv_sec", "tv_usec", "pkt_nsec", "pkt_num"]
            # Check if module or quabo
            if "quabo_0.tv_sec" in seq.metadata_offsets:
                keys = [f"quabo_0.{k}" for k in keys]
        
        # Filter keys to those actually present
        keys = [k for k in keys if k in seq.metadata_offsets]
        n_frames = len(seq)
        
        t0 = time.perf_counter()
        _ = seq.get_metadata_arrays(keys)
        elapsed = time.perf_counter() - t0
        
        return self._record_result("Metadata Extraction", product_name, n_frames, elapsed, seq, {"n_keys": len(keys)})

    def _record_result(self, test_name: str, product_name: str, n_frames: int, elapsed: float, seq: PFFSequence, extra: dict[str, Any] | None = None) -> ProfileResult:
        fps = n_frames / elapsed if elapsed > 0 else 0
        
        # Calculate bytes processed
        bytes_per_frame = seq.frame_config.frame_size if seq.frame_config else 0
        total_mb = (n_frames * bytes_per_frame) / (1024 * 1024)
        throughput = total_mb / elapsed if elapsed > 0 else 0
        
        res = ProfileResult(
            test_name=test_name,
            product_name=product_name,
            total_frames=n_frames,
            elapsed_seconds=elapsed,
            fps=fps,
            throughput_mbs=throughput,
            extra_meta=extra or {}
        )
        self.results.append(res)
        return res

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def display_results(self) -> None:
        console = Console()
        table = Table(title=f"Performance Profile: {self.run.run_dir.name}")
        
        table.add_column("Test", style="cyan")
        table.add_column("Product", style="magenta")
        table.add_column("Frames", justify="right")
        table.add_column("Elapsed (s)", justify="right")
        table.add_column("FPS", justify="right", style="green")
        table.add_column("Throughput (MB/s)", justify="right", style="bold yellow")

        for r in self.results:
            table.add_row(
                r.test_name,
                r.product_name,
                f"{r.total_frames:,}",
                f"{r.elapsed_seconds:.3f}",
                f"{r.fps:,.1f}",
                f"{r.throughput_mbs:,.1f}"
            )
        
        console.print(table)

    def plot_results(self) -> None:
        """
        Visual display of profile outputs using matplotlib/seaborn.
        Useful for Jupyter notebooks.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not self.results:
            print("No results to plot.")
            return

        df = self.get_results_df()
        
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.barplot(data=df, x="test_name", y="fps", hue="product_name", ax=ax1)
        ax1.set_title("Frames Per Second (FPS)")
        ax1.set_ylabel("FPS")
        ax1.tick_params(axis='x', rotation=45)

        sns.barplot(data=df, x="test_name", y="throughput_mbs", hue="product_name", ax=ax2)
        ax2.set_title("I/O Throughput (MB/s)")
        ax2.set_ylabel("MB/s")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
