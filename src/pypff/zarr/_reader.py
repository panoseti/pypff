"""
Read-side wrappers for PanoSETI Zarr v3 stores.

Mirrors the pypff.io2.PanosetiRun / PFFSequence API so users have one mental
model whether they are reading source PFF or converted Zarr.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np


class PanosetiZarrStore:
    """Read-side wrapper for one converted (data_product, module) Zarr store.

    Provides typed access to images, timestamps, and header variables without
    requiring users to know the xarray or zarr-python API details.

    For direct xarray access::

        ds = store.to_dataset()          # xarray.Dataset
        ds.images                         # (T, H, W) DataArray
        ds.unix_t_ns                      # int64 timestamp DataArray
        ds.pkt_num                        # uint32 header DataArray
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Zarr store not found: {self.path}")

    @cached_property
    def _attrs(self) -> dict[str, Any]:
        import zarr
        g = zarr.open_group(str(self.path), mode="r")
        return dict(g.attrs)

    @cached_property
    def _dataset(self) -> Any:  # xarray.Dataset
        import xarray as xr
        # consolidated=False because we don't write consolidated metadata
        # (it's non-spec for Zarr v3 and emits a ZarrUserWarning).
        return xr.open_zarr(str(self.path), consolidated=False)

    # ── core metadata ────────────────────────────────────────────────────────

    @property
    def data_product(self) -> str:
        return str(self._attrs.get("data_product", "unknown"))

    @property
    def module(self) -> str:
        return str(self._attrs.get("module", "unknown"))

    @property
    def bytes_per_pixel(self) -> int:
        return int(self._attrs.get("bytes_per_pixel", 0))

    @property
    def total_frames(self) -> int:
        return int(self._attrs.get("total_frames", 0))

    @property
    def header_format(self) -> str:
        return str(self._attrs.get("header_format", "single"))

    @property
    def header_fields(self) -> list[str]:
        """Root-level (non-quabo) header array names."""
        return list(self._attrs.get("header_fields", []))

    @property
    def quabo_fields(self) -> list[str]:
        """Module-level quabo header array names (e.g. quabo_0_pkt_num)."""
        return list(self._attrs.get("quabo_fields", []))

    @property
    def source_pff_files(self) -> list[str]:
        return list(self._attrs.get("source_pff_files", []))

    @property
    def frame_config(self) -> dict[str, Any]:
        """Raw frame config dict as stored in root attrs."""
        return dict(self._attrs.get("frame_config", {}))

    @property
    def run_configs(self) -> dict[str, Any]:
        """Parsed run configs embedded at conversion time (may be empty)."""
        return dict(self._attrs.get("run_configs", {}))

    # ── data access ──────────────────────────────────────────────────────────

    def to_dataset(self) -> Any:  # xarray.Dataset
        """Return an xarray.Dataset with all arrays (images, timestamps, headers)."""
        return self._dataset

    @property
    def images(self) -> Any:  # xarray.DataArray
        """(T, H, W) pixel data as a dask-backed xarray DataArray."""
        return self._dataset["images"]

    def timestamps(self, *, as_datetime: bool = False) -> np.ndarray:
        """Return timestamps as int64 nanoseconds, or as datetime64[ns] view."""
        ts_ns: np.ndarray = self._dataset["unix_t_ns"].values
        return ts_ns.view("datetime64[ns]") if as_datetime else ts_ns

    def __len__(self) -> int:
        return self.total_frames

    def __repr__(self) -> str:
        return (
            f"PanosetiZarrStore("
            f"dp={self.data_product!r}, module={self.module!r}, "
            f"T={self.total_frames:,}, path={self.path.name!r})"
        )


class PanosetiZarrRun:
    """Read-side wrapper for a converted run output directory.

    Discovers ``.zarr`` stores and a sibling ``.panoseti-meta/`` sidecar bundle
    written by :func:`pypff.zarr.convert_run`.

    Example::

        from pypff.zarr import PanosetiZarrRun

        zrun = PanosetiZarrRun("output/L0_zarr")
        for name in zrun.list_products():
            store = zrun.get_product(name)
            ds = store.to_dataset()
            print(name, ds.images.shape)

        print(zrun.configs.get("obs_config"))
    """

    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.out_dir}")

    @cached_property
    def _store_map(self) -> dict[str, PanosetiZarrStore]:
        """Discover all .zarr stores and index them by product name."""
        stores: dict[str, PanosetiZarrStore] = {}
        for zarr_path in sorted(self.out_dir.glob("*.zarr")):
            if not zarr_path.is_dir():
                continue
            store = PanosetiZarrStore(zarr_path)
            # Derive a canonical product key from authoritative root attrs
            dp = store.data_product
            bpp = store.bytes_per_pixel
            mod = store.module
            key = f"dp_{dp}.bpp_{bpp}.module_{mod}"
            stores[key] = store
        return stores

    @cached_property
    def _meta_dir(self) -> Path | None:
        """Locate the .panoseti-meta/ sidecar directory if present."""
        candidates = sorted(self.out_dir.glob("*.panoseti-meta"))
        return candidates[0] if candidates else None

    def list_products(self) -> list[str]:
        """Return sorted canonical product names (e.g. 'dp_ph256.bpp_2.module_254')."""
        return sorted(self._store_map.keys())

    def get_product(self, name: str) -> PanosetiZarrStore:
        """Return the :class:`PanosetiZarrStore` for *name*."""
        if name not in self._store_map:
            available = list(self._store_map.keys())
            raise KeyError(f"Product {name!r} not found. Available: {available}")
        return self._store_map[name]

    @cached_property
    def configs(self) -> dict[str, Any]:
        """
        Parsed run configs, sourced from (in priority order):

        1. The ``.panoseti-meta/configs/`` sidecar directory (raw JSON files).
        2. The ``run_configs`` root attrs embedded in any one product store.
        """
        # Try sidecar directory first
        if self._meta_dir is not None:
            cfg_dir = self._meta_dir / "configs"
            if cfg_dir.is_dir():
                result: dict[str, Any] = {}
                for cfg_file in sorted(cfg_dir.glob("*.json")):
                    try:
                        data = json.loads(cfg_file.read_text())
                        # strip .json extension for the key
                        result[cfg_file.stem] = data
                    except Exception:
                        pass
                if result:
                    return result

        # Fall back to embedded run_configs in the first store
        for store in self._store_map.values():
            rc = store.run_configs
            if rc:
                return rc
        return {}

    def list_sidecars(self) -> list[Path]:
        """Return all files in the .panoseti-meta/ bundle (flat list)."""
        if self._meta_dir is None:
            return []
        return sorted(f for f in self._meta_dir.rglob("*") if f.is_file())

    def get_log(self, name: str) -> str:
        """Read a log file from the sidecar bundle by filename."""
        if self._meta_dir is None:
            raise FileNotFoundError("No .panoseti-meta/ sidecar directory found.")
        log_path = self._meta_dir / "logs" / name
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {name}")
        return log_path.read_text()

    def __repr__(self) -> str:
        n = len(self._store_map)
        return f"PanosetiZarrRun(out_dir={self.out_dir.name!r}, {n} product(s))"


def open_zarr_run(path: str | Path) -> PanosetiZarrRun:
    """Open a converted run output directory as a :class:`PanosetiZarrRun`."""
    return PanosetiZarrRun(path)
