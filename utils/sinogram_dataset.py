"""
SinogramDataset
===============
Lazy-loading dataset of paired (noisy, FD) fan-beam sinogram slices built from
the manifest produced by data_prep/sinogram_data_prep.py.

Each sample is one axial (z) slice  →  one 2-D fan-beam sinogram of shape
    (num_angles [after optional subsampling], num_detectors)

The TIFF layout saved by helix2fan is (rotview, nu, nz_rebinned):
  axis-0  – angular views  (rotview per rotation)
  axis-1  – detector columns (nu transverse elements)
  axis-2  – axial slices (nz_rebinned)

So the 2-D sinogram for slice z is  arr[:, :, z]  →  (rotview, nu).

Geometry (sdd, sid, du, etc.) is read once from the first FD TIFF's metadata
and stored as  dataset.geometry  for use by DifferentiableFBP.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# helix2fan metadata helper (reads ImageDescription JSON from OME-TIFF)
# ---------------------------------------------------------------------------
_HELIX2FAN_DIR = Path(__file__).parent.parent / "extern" / "helix2fan"
if str(_HELIX2FAN_DIR) not in sys.path:
    sys.path.insert(0, str(_HELIX2FAN_DIR))

try:
    from helper import load_tiff_stack_with_metadata as _load_meta  # noqa: E402
    _HELIX2FAN_OK = True
except ImportError:
    _HELIX2FAN_OK = False

# Default geometry for Mayo LDCT scanner (fallback if metadata is absent)
_DEFAULT_GEO = {
    "nu": 736,
    "du": 1.2858,
    "dso": 595.0,
    "dsd": 1085.6,
    "rotview": 2304,
    "det_central_element": [367.5, 0.0],
}


def _read_geometry(tif_path: Path) -> dict:
    """Read scanner geometry from TIFF metadata; fall back to defaults."""
    if _HELIX2FAN_OK:
        try:
            _, meta = _load_meta(tif_path)
            if meta:
                return meta
        except Exception:
            pass
    return _DEFAULT_GEO.copy()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SinogramDataset(Dataset):
    """Paired (noisy, FD) 2-D fan-beam sinogram slices.

    Parameters
    ----------
    manifest_path : str or Path
        Path to the manifest.json produced by projection_data_prep.py.
    root_dir : str or Path
        Project root so that relative paths in the manifest resolve correctly.
    mode : {'train', 'val', 'test'}
    val_id : str
        Scan ID reserved for validation (e.g. 'L310').
    test_id : str
        Scan ID reserved for testing (e.g. 'L333').
    angle_subsample : int
        Take every N-th projection angle.  1 = no subsampling.
    norm_mean, norm_std : float or None
        Pre-computed normalisation statistics.  When None, normalisation is
        skipped.  Compute them from the training split first, then pass to
        val/test datasets for consistency.

    Attributes
    ----------
    geometry : dict
        Scanner geometry extracted from TIFF metadata (sdd, sid, du, …).
    norm_mean, norm_std : float or None
        Normalisation statistics used by this dataset.
    """

    def __init__(
        self,
        manifest_path,
        root_dir,
        mode="train",
        val_id="L310",
        test_id="L333",
        angle_subsample=1,
        norm_mean=None,
        norm_std=None,
    ):
        self.angle_subsample = angle_subsample
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        manifest_path = Path(manifest_path)
        root = Path(root_dir)

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Build index and path table
        self.index_map = []          # [(scan_id, z), ...]
        self.scan_paths = {}         # scan_id -> (noisy_path_str, fd_path_str)
        self.geometry = None
        geo_loaded = False

        for scan in manifest["scans"]:
            sid = scan["scan_id"]

            if mode == "train" and sid in (val_id, test_id):
                continue
            if mode == "val" and sid != val_id:
                continue
            if mode == "test" and sid != test_id:
                continue

            fd_path = root / scan["fd_tif"]
            noisy_path = root / scan["noisy_tif"]
            num_slices = scan["shape"][2]  # nz_rebinned

            self.scan_paths[sid] = (str(noisy_path), str(fd_path))

            for z in range(num_slices):
                self.index_map.append((sid, z))

            if not geo_loaded:
                self.geometry = _read_geometry(fd_path)
                geo_loaded = True

        if not self.index_map:
            raise ValueError(
                f"No scans found for mode='{mode}' "
                f"(val_id='{val_id}', test_id='{test_id}').\n"
                f"Available scan IDs: {[s['scan_id'] for s in manifest['scans']]}"
            )

        if self.geometry is None:
            self.geometry = _DEFAULT_GEO.copy()

        # In-process cache: scan_id -> (noisy_arr, fd_arr) numpy float32
        # Loaded on first access; efficient for num_workers=0.
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def compute_norm_stats(self, max_scans=4, pixel_stride=16):
        """Estimate (mean, std) from FD sinograms in this dataset.

        Samples every `pixel_stride`-th value from at most `max_scans` scans
        for speed.  Call on the training dataset before constructing val/test.

        Returns
        -------
        mean : float
        std  : float
        """
        vals = []
        for sid in list(self.scan_paths.keys())[:max_scans]:
            _, fd_arr = self._get_scan_arrays(sid)
            vals.append(fd_arr.ravel()[::pixel_stride])
        combined = np.concatenate(vals)
        return float(np.mean(combined)), float(np.std(combined))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_scan_arrays(self, scan_id):
        """Return (noisy_arr, fd_arr) for the given scan, loading once."""
        if scan_id not in self._cache:
            noisy_path, fd_path = self.scan_paths[scan_id]
            noisy_arr = tifffile.imread(noisy_path).astype(np.float32)
            fd_arr = tifffile.imread(fd_path).astype(np.float32)
            self._cache[scan_id] = (noisy_arr, fd_arr)
        return self._cache[scan_id]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        scan_id, z = self.index_map[idx]
        noisy_arr, fd_arr = self._get_scan_arrays(scan_id)

        # 2-D slice: (rotview, nu)
        noisy = noisy_arr[:, :, z].copy()
        fd = fd_arr[:, :, z].copy()

        # Angle subsampling
        if self.angle_subsample > 1:
            noisy = noisy[:: self.angle_subsample]
            fd = fd[:: self.angle_subsample]

        # Normalise to zero-mean / unit-std if stats are available
        if self.norm_mean is not None and self.norm_std is not None:
            scale = self.norm_std + 1e-8
            noisy = (noisy - self.norm_mean) / scale
            fd = (fd - self.norm_mean) / scale

        # Add channel dim → (1, num_angles, num_detectors)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        fd_t = torch.from_numpy(fd).unsqueeze(0)

        return noisy_t, fd_t
