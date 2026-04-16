"""
LoDoPaBDataset
==============
PyTorch Dataset for the LoDoPaB-CT dataset, loaded directly from HDF5 files
without any dival/odl dependency.

Reference
---------
  Leuschner, Schmidt, Baguer & Maass (2021).
  "LoDoPaB-CT, a benchmark dataset for low-dose computed tomography
  reconstruction."  Scientific Data 8:109.
  https://doi.org/10.1038/s41597-021-00893-z

Each sample is a paired (noisy_sinogram, clean_gt_image):
  - noisy_sinogram : Tensor (1, 1000, 513)  — post-log sinogram normalised by μ_max
  - gt_image       : Tensor (1, 362, 362)   — clean CT image normalised by μ_max

Value ranges
------------
  Both sinogram and GT image are stored divided by
  MU_MAX = 81.35858 m⁻¹ (Eq. 10 of the paper), so values lie in [0, 1].
  To recover physical attenuation coefficients (m⁻¹): multiply by MU_MAX.
  Recommended display window for GT images: [0, 0.45]  (≈ HU range [−1001, 831],
  matching Fig. 5 of the paper).

GT orientation / transpose convention
--------------------------------------
  Per Fig. 4 (line 3) of the paper, ground truth images are stored TRANSPOSED
  relative to the standard (row = y, col = x) image convention:
      ẑ = transpose(center_crop(z, 362 × 362))
  This converts DICOM row-major layout to ODL's (x, y) column-major layout.
  To display or compare with standard-orientation images:
      gt_display = gt_np.T          # numpy
      gt_display = gt_tensor[0].T   # torch

Noise model
-----------
  Sinograms contain simulated low-dose Poisson noise:
      Ñᵢ ~ Pois(N0 · exp(−𝒜x))
  with N0 = 4096 photons/pixel (≈ 1/10 of clinical ~40 000).
  Stored sinogram: y = −ln(Ñ/N0) / μ_max.
  There are NO beam-hardening, scatter, or ring artifacts in the base dataset.

Optional synthetic artifacts (add_artifacts=True)
-------------------------------------------------
  Three additional degradations can be injected on-the-fly in __getitem__
  (not part of the original paper's dataset):
    • motion_blur  — Gaussian blur along the angle axis; simulates patient motion.
    • ring         — multiplicative gain error on random detector columns (all
                     angles); simulates miscalibrated detector elements.  Appears
                     as bright/dark concentric rings after FBP reconstruction.
    • metal        — sinusoidal high-attenuation trace; simulates a dense metal
                     implant.  Causes severe streak artifacts after FBP.

  Artifacts are injected in the μ_max-normalised sinogram scale [0, 1].
  After injection the sinogram is clipped to [0, 1] to suppress physically
  implausible values introduced by the multiplicative ring/metal models.
  No z-score normalisation is applied; the physical [0, 1] scale is kept
  throughout, consistent with the official LoDoPaB-CT challenge baselines.

HDF5 layout (produced by data_prep/download_lodopab.py)
--------------------------------------------------------
  observation_{part}_{idx:03d}.hdf5   — key 'data', shape (128, 1000, 513), float32
  ground_truth_{part}_{idx:03d}.hdf5  — key 'data', shape (128, 362, 362),  float32

Geometry (Leuschner et al. 2021, Methods section)
--------------------------------------------------
  Parallel beam, 1000 angles uniformly in [0, π),
  513 detector elements, image 362 × 362 pixels,
  physical domain 26 cm × 26 cm.
"""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_ANGLES           = 1000
NUM_DET_PIXELS       = 513
IMAGE_SIZE           = 362          # height = width
NUM_SAMPLES_PER_FILE = 128

# Physical constants from the paper (Eq. 10, Fig. 4)
MU_MAX              = 81.35858      # m⁻¹  — normalisation constant (largest representable HU value)
N0                  = 4096          # mean photon count without attenuation (≈ 1/10 clinical dose)
PHYSICAL_DOMAIN_M   = 0.26          # metres — side length of the square image domain (26 cm)

LEN = {
    "train":      35820,
    "validation": 3522,
    "test":       3553,
}

# ── Default data path — must match DATA_PATH in data_prep/download_lodopab.py ──
DEFAULT_DATA_PATH = Path("/home/shiblu/Project/Masters_thesis/Masters-Thesis/medical_image_datasets/lodopab")

# ── Default artifact configuration ────────────────────────────────────────────

DEFAULT_ARTIFACT_CONFIG = {
    # Motion blur: Gaussian σ in number of projection angles along the angle axis.
    # σ = 1–3 rows corresponds to mild-to-moderate patient motion in a 1000-angle
    # parallel-beam geometry (~0.18°–0.54° arc per row).  No direct benchmark exists
    # for this parameterisation; the range is consistent with the "small motion"
    # regime described in:
    #   Haggag et al. (2025), arXiv:2509.10961 — HR-pQCT motion simulation study.
    # p=0.5: applied to half of all samples so the network also trains on
    # motion-free sinograms; independent of other artifact types.
    "motion_blur":        True,
    "motion_blur_prob":   0.5,           # per-sample Bernoulli probability
    "motion_blur_sigma":  (1.0, 3.0),   # (min, max) σ in projection rows; sampled uniformly per item

    # Ring artifact: multiplicative per-column gain error applied to all angles.
    # Models detector element miscalibration (gain inhomogeneity).
    # Gain factor ∈ [0.90, 1.10] corresponds to ±10% gain error — the standard
    # "realistic" value for medical CT detector miscalibration, established in:
    #   An et al. (2020), cited as the canonical ring-artifact reference in
    #   Liang et al. (2025), "Improving Artifact Robustness for CT Deep Learning
    #   Models Without Labeled Artifact Images via Domain Adaptation",
    #   arXiv:2510.06584.
    # Number of affected columns: 1–5 (sparse point failures).  The SynthRAR paper
    # (Zeng et al., 2025, arXiv:2602.11880) uses 2% of all detectors (~10 columns
    # for 513 elements) for complete column failures; 1–5 is a conservative subset.
    # p=0.5: ring artifacts are scanner-hardware-specific, not present in every scan.
    "ring":               True,
    "ring_prob":          0.5,           # per-sample Bernoulli probability
    "ring_n_cols":        (1, 5),        # (min, max) number of affected columns
    "ring_strength":      (0.90, 1.10),  # multiplicative gain factor (±10%)

    # Metal artifact: dense implant sinusoidal trace.
    # The bin multiplier models the ratio of metal-to-tissue linear attenuation.
    # At 100 keV (120 kVp polychromatic beam), NIST/XCOM tables give:
    #   Titanium : μ ≈ 1.22 cm⁻¹  (μ/ρ = 0.272 cm²/g × 4.506 g/cm³)
    #   Water    : μ ≈ 0.171 cm⁻¹
    #   Ratio    : ~7×  (iron ≈ 5–6×; gold ≈ 25–30×)
    # Scale range [4, 8] covers iron through titanium implants.  Reference:
    #   Wang et al. (2018), "Convolutional Neural Network based Metal Artifact
    #   Reduction in X-ray Computed Tomography", IEEE TMI, PMC:5998663.
    # p=0.3: metal implants are patient-specific and less common; lower probability
    # prevents the severe saturation from dominating the training distribution.
    "metal":              True,
    "metal_prob":         0.3,           # per-sample Bernoulli probability
    "metal_n_objects":    (1, 2),        # (min, max) number of implants
    "metal_width":        (4, 12),       # detector-bin width of the metal shadow
    "metal_scale":        (4.0, 8.0),    # attenuation multiplier (4× iron – 8× titanium)
}


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _motion_blur(sino: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur along the angle axis (axis 0)."""
    return gaussian_filter1d(sino, sigma=sigma, axis=0).astype(np.float32)


def _ring_artifact(sino: np.ndarray, n_cols: int, gain_range: tuple) -> np.ndarray:
    """Apply multiplicative gain error to random detector columns (all angles).

    Each selected column is multiplied by a gain factor drawn uniformly from
    `gain_range`, e.g. (0.90, 1.10) for ±10% miscalibration.  The constant
    column-wise scaling is the standard ring-artifact model used in the
    literature (An et al. 2020; arXiv:2510.06584).
    """
    sino = sino.copy()
    cols = np.random.choice(NUM_DET_PIXELS, size=n_cols, replace=False)
    for col in cols:
        gain = np.random.uniform(*gain_range)
        sino[:, col] *= gain
    return sino


def _metal_artifact(
    sino: np.ndarray,
    n_objects: int,
    width: int,
    scale: float,
) -> np.ndarray:
    """Sinusoidal high-attenuation trace simulating a metal implant.

    A metal object at image-space position (x₀, y₀) casts its shadow onto
    detector bin  t(θ) = x₀·cos θ + y₀·sin θ  at each projection angle θ.
    The bins inside the shadow are multiplied by `scale`.
    """
    sino   = sino.copy()
    angles = np.linspace(0.0, math.pi, NUM_ANGLES, endpoint=False)
    center = NUM_DET_PIXELS // 2

    for _ in range(n_objects):
        # Random metal object position in pixel coordinates (image-space)
        radius = np.random.uniform(0.05, 0.35) * IMAGE_SIZE
        theta0 = np.random.uniform(0, 2 * math.pi)
        x0     = radius * math.cos(theta0)
        y0     = radius * math.sin(theta0)
        w      = width if isinstance(width, int) else np.random.randint(*width)
        s      = scale if isinstance(scale, (int, float)) else np.random.uniform(*scale)

        # Detector position of the shadow at each angle — shape (NUM_ANGLES,)
        det_pos = (x0 * np.cos(angles) + y0 * np.sin(angles) + center).astype(int)

        for ai in range(NUM_ANGLES):
            d       = det_pos[ai]
            d_start = max(0, d - w // 2)
            d_end   = min(NUM_DET_PIXELS, d + w // 2 + 1)
            if d_start < d_end:
                sino[ai, d_start:d_end] *= s

    return sino


# ── Dataset ───────────────────────────────────────────────────────────────────

class LoDoPaBDataset(Dataset):
    """Paired (noisy sinogram, clean image) dataset for LoDoPaB-CT.

    Parameters
    ----------
    mode : {'train', 'validation', 'test'}
    data_path : str or Path, optional
        Root directory containing the HDF5 files.  Defaults to
        DEFAULT_DATA_PATH (must match DATA_PATH in download_lodopab.py).
    add_artifacts : bool
        When True, randomly inject motion blur, ring, and/or metal artifacts
        into each sinogram.  After injection the sinogram is clipped to [0, 1]
        to suppress physically implausible values.  Controlled by
        ``artifact_config``.  Default: False.
    artifact_config : dict or None
        Override individual artifact parameters.  Missing keys fall back to
        DEFAULT_ARTIFACT_CONFIG.  Pass e.g.
        ``{"ring": False}`` to disable ring artifacts while keeping the rest.

    Attributes
    ----------
    geometry : dict
        Parallel-beam geometry parameters for DifferentiableParallelFBP.

    Notes
    -----
    Sinograms are returned in the physical [0, 1] scale (divided by
    MU_MAX = 81.35858 m⁻¹ as stored in the HDF5 files).  No z-score
    normalisation is applied; this is consistent with all official
    LoDoPaB-CT challenge baselines (Leuschner et al. 2021).
    """

    # Parallel-beam geometry (Leuschner et al. 2021, Methods section)
    GEOMETRY = {
        "num_angles":       NUM_ANGLES,
        "num_detectors":    NUM_DET_PIXELS,
        "image_size":       IMAGE_SIZE,
        # Detector spacing in pixel units: image diagonal / num_detectors ≈ 0.996
        "detector_spacing": IMAGE_SIZE * math.sqrt(2) / NUM_DET_PIXELS,
        "voxel_spacing":    1.0,
        # Physical scale
        "mu_max":           MU_MAX,           # m⁻¹ — multiply normalised values by this
        "n0":               N0,               # mean photon count (noise level)
        "physical_domain_m": PHYSICAL_DOMAIN_M,  # 26 cm × 26 cm square domain
    }

    def __init__(
        self,
        mode:            str  = "train",
        data_path             = None,
        add_artifacts:   bool = False,
        artifact_config: dict = None,
    ):
        if mode not in LEN:
            raise ValueError(f"mode must be one of {list(LEN)}, got '{mode}'")

        self.mode          = mode
        self.data_path     = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.add_artifacts = add_artifacts
        self.geometry      = self.GEOMETRY.copy()
        self._length       = LEN[mode]

        # Merge user overrides into defaults
        self._artifact_cfg = {**DEFAULT_ARTIFACT_CONFIG, **(artifact_config or {})}

        # Validate that at least the first file exists
        first_obs = self.data_path / f"observation_{mode}_000.hdf5"
        if not first_obs.exists():
            raise FileNotFoundError(
                f"LoDoPaB data not found at {self.data_path}\n"
                f"Run:  conda run -n tensor python data_prep/download_lodopab.py"
            )

    # ── Artifact injection ────────────────────────────────────────────────────

    def _apply_artifacts(self, sino: np.ndarray) -> np.ndarray:
        """Apply enabled artifacts to a single sinogram (1000, 513).

        Each artifact type is gated by an independent Bernoulli draw using its
        ``*_prob`` key, so the network trains on all combinations (none / any
        subset / all three) rather than always seeing the full degradation.
        """
        cfg = self._artifact_cfg

        if cfg.get("motion_blur", False) and np.random.rand() < cfg.get("motion_blur_prob", 1.0):
            lo, hi = cfg["motion_blur_sigma"]
            sigma  = np.random.uniform(lo, hi)
            sino   = _motion_blur(sino, sigma)

        if cfg.get("ring", False) and np.random.rand() < cfg.get("ring_prob", 1.0):
            n_lo, n_hi = cfg["ring_n_cols"]
            n_cols     = np.random.randint(n_lo, n_hi + 1)
            sino       = _ring_artifact(sino, n_cols, cfg["ring_strength"])

        if cfg.get("metal", False) and np.random.rand() < cfg.get("metal_prob", 1.0):
            n_lo, n_hi = cfg["metal_n_objects"]
            n_obj      = np.random.randint(n_lo, n_hi + 1)
            sino       = _metal_artifact(
                sino,
                n_objects = n_obj,
                width     = cfg["metal_width"],
                scale     = cfg["metal_scale"],
            )

        return sino

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"index {idx} out of range for '{self.mode}' ({self._length})")

        file_idx    = idx // NUM_SAMPLES_PER_FILE
        idx_in_file = idx  % NUM_SAMPLES_PER_FILE

        obs_path = self.data_path / f"observation_{self.mode}_{file_idx:03d}.hdf5"
        gt_path  = self.data_path / f"ground_truth_{self.mode}_{file_idx:03d}.hdf5"

        with h5py.File(obs_path, "r") as f:
            sino = f["data"][idx_in_file].astype(np.float32)   # (1000, 513)

        with h5py.File(gt_path, "r") as f:
            img  = f["data"][idx_in_file].astype(np.float32)   # (362, 362)

        # Inject artifacts and clip to physical [0, 1] range.
        # Clipping is necessary because the multiplicative ring model and the
        # metal scale can push a small number of bins slightly outside [0, 1].
        # The dataset is already normalised by MU_MAX, so [0, 1] is the correct
        # physical range; no z-score shift is applied.
        if self.add_artifacts:
            sino = self._apply_artifacts(sino)
            sino = np.clip(sino, 0.0, 1.0)

        sino_t = torch.from_numpy(sino).unsqueeze(0)   # (1, 1000, 513)
        img_t  = torch.from_numpy(img ).unsqueeze(0)   # (1, 362, 362)

        return sino_t, img_t
