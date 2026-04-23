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
    • motion_blur  — periodic lateral shift of detector data (Lu & Mackie 2002;
                     Mao et al. 2007); simulates respiratory patient motion.
    • ring         — additive post-log offset on random detector columns,
                     equivalent to a per-column calibration/gain error
                     (Sijbers & Postnov 2004; Prell et al. 2009).  Appears as
                     concentric rings after FBP reconstruction.
    • metal        — image-domain implant mask forward-projected (offline, once,
                     via scripts/build_metal_library.py) and added to the
                     sinogram, followed by photon-starvation clipping at the
                     detector noise floor (Gjesteby et al. 2016; Boas &
                     Fleischmann 2012).  Produces the characteristic bright/dark
                     saturation streaks.

  Artifacts are injected in the μ_max-normalised sinogram scale [0, 1].
  After injection the sinogram is clipped to [0, 1] to suppress physically
  implausible values introduced by the ring/metal models.
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
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import shift as ndimage_shift
from torch.utils.data import Dataset

# Default location of the metal-implant library produced by
# scripts/build_metal_library.py.  Can be overridden per-instance.
DEFAULT_METAL_LIBRARY = (
    Path(__file__).resolve().parents[1] / "data" / "metal" / "metal_library.npz"
)

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

# ── Default data path — override with LODOPAB_DATA_PATH env var on remote machines ──
DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "LODOPAB_DATA_PATH",
        Path(__file__).parent.parent / "medical_image_datasets" / "lodopab",
    )
)

# ── Default artifact configuration ────────────────────────────────────────────

DEFAULT_ARTIFACT_CONFIG = {
    # Motion artifact: periodic lateral shift of the detector data.
    #
    # Physical model:
    #   In parallel-beam CT, a patient translation Δx at projection angle θ
    #   shifts the sinogram radially:  t → t − Δx·cosθ − Δy·sinθ.
    #   This is a fundamental property of the Radon transform: any point
    #   (x₀, y₀) in the image traces a sinusoid in the (θ, t) sinogram.
    #   For periodic respiratory motion the detector-axis displacement is:
    #       δt(θ) = A · sin(2π · f · θ / N_angles + φ)
    #   where A is the amplitude in detector bins and f is the number of
    #   breathing cycles during the scan.  After FBP, this produces the
    #   characteristic ghosting and edge-blurring seen in clinical chest/
    #   abdominal CT with patient motion.
    #
    #   References:
    #     Lu & Mackie (2002), "Tomographic motion detection and correction
    #       directly in sinogram space", Phys. Med. Biol. 47(8):1267–1284.
    #       DOI: 10.1088/0031-9155/47/8/304
    #       [foundational derivation of the t → t − Δx·cosθ shift model]
    #     Lu et al. (2005), "Reduction of motion blurring artifacts using
    #       respiratory gated CT in sinogram space: a quantitative evaluation",
    #       Med. Phys. 32(11):3295–3304.  DOI: 10.1118/1.2074187
    #       [quantifies respiratory motion ±5–15 mm in abdominal CT]
    #     Mao et al. (2007), "CT image registration in sinogram space",
    #       Med. Phys. 34(9):3596–3602.  DOI: 10.1118/1.2767402
    #       [explicit treatment of rigid translation → sinogram shift]
    #
    # Parameters:
    #   motion_amplitude : (min, max) peak shift in detector bins.
    #     LoDoPaB detector spacing ≈ 0.72 mm/bin.  10–20 bins ≈ 7–14 mm,
    #     consistent with moderate respiratory motion in chest/abdominal CT
    #     (Lu et al. 2005 report ±5–15 mm for abdominal CT).  Amplitudes
    #     below ~10 bins are masked by the existing Poisson noise floor.
    #   motion_cycles : (min, max) breathing cycles during the scan.
    #     0.5–1.5 cycles produces coherent blurring and double contours
    #     (the "unsharpness" and "double cortices" described in Lu et al.
    #     2005).  More than ~2 cycles creates rapid oscillations that cause
    #     aliasing streaks rather than smooth motion blur.
    #   p=0.5: applied to half of all samples so the network also trains on
    #   motion-free sinograms.
    "motion_blur":        True,
    "motion_blur_prob":   0.5,
    "motion_amplitude":   (10, 20),      # peak shift in detector bins (≈7–14 mm)
    "motion_cycles":      (0.5, 1.5),   # breathing cycles during scan

    # Ring artifact: additive per-column offset applied to all angles.
    #
    # Physical model
    # --------------
    # A detector-element calibration error (dark-current offset, non-uniform
    # gain, or damaged element) produces a multiplicative photon-count error:
    #     N_obs = g · N_true
    # After the post-log transform used by LoDoPaB (y = −ln(N/N0)/μ_max),
    # this multiplicative error becomes a column-constant *additive* offset:
    #     y_obs(θ, t_c) = y_true(θ, t_c) + c_c,
    #     c_c = −ln(g_c) / μ_max.
    # The same additive-offset ring model is used in Sijbers & Postnov (2004)
    # Phys. Med. Biol. 49:N247 and Prell et al. (2009) Phys. Med. Biol.
    # 54:3881, and is the canonical form for learned ring removal in
    # Liang et al. (2025), arXiv:2510.06584 and the SynthRAR protocol of
    # Zeng et al. (2025), arXiv:2602.11880.  A constant column offset maps
    # to a concentric ring after FBP back-projection.
    #
    # Parameterisation: we sample the physical gain g_c directly from
    # ring_gain = (g_lo, g_hi) with 0 < g_lo < g_hi < 1 (under-sensitive
    # detectors → bright rings).  With probability 0.5 the element is modelled
    # as over-sensitive by inverting g → 1/g (dark ring), keeping the
    # perturbation symmetric in log-gain space.  The additive post-log offset
    # is then c_c = −ln(g_c)/μ_max.  Default gain_range=(0.5, 0.95) spans
    # realistic 5–50 % gain errors, corresponding to |c_c| ∈ [6.3e-4, 8.5e-3]
    # — 2–13 × smaller than the signal-driven ±0.02 used in earlier versions
    # of this code, and consistent with the fractional gain errors reported
    # in Sijbers & Postnov (2004) and Prell et al. (2009) for partially
    # damaged and drift-affected detector elements.
    # Number of affected columns 1–5 out of 513 models sparse point-detector
    # failures (SynthRAR uses ≈2% column failure rate; our 1–5 is a
    # conservative subset).
    "ring":               True,
    "ring_prob":          0.5,             # per-sample Bernoulli probability
    "ring_n_cols":        (1, 5),          # (min, max) number of affected columns
    "ring_gain":          (0.5, 0.95),     # detector gain range (under-sensitive; inverted with p=0.5)

    # Metal artifact: image-domain mask → forward projection → additive
    # line-integral contribution → photon-starvation clipping.
    #
    # Physical model
    # --------------
    # A dense implant adds its attenuation line integral s_metal(θ, t) to the
    # tissue line integral (monoenergetic Beer–Lambert regime):
    #     y_total(θ, t) = y_tissue(θ, t) + s_metal(θ, t).
    # The sinusoidal shadow trace s_metal is obtained by forward-projecting
    # an image-domain metal mask through the LoDoPaB parallel-beam geometry
    # (done once offline by scripts/build_metal_library.py).  At runtime we
    # pick a precomputed centred-mask sinogram from the library and translate
    # it by (x0, y0) using the Radon shift identity t → t + x0·cos θ + y0·sin θ
    # (Lu & Mackie 2002; Mao et al. 2007) — avoiding per-sample forward
    # projection at training time.
    #
    # After adding the metal contribution we apply a photon-starvation clip:
    #     N = N0 · exp(−y · μ_max),   N ← max(N, 1),
    #     y ← −ln(N/N0) / μ_max.
    # This enforces the detector noise floor (at most log(N0) ≈ 8.3 / μ_max
    # ≈ 0.102 in normalised units per ray).  Rays that would count <1 photon
    # through the implant become the characteristic saturation streaks seen
    # in clinical metal-artifact CT.  See Gjesteby et al. (2016), IEEE Access
    # 4:5826 and Boas & Fleischmann (2012), Imaging Med. 4:229 for the
    # underlying physics.
    #
    # metal_scale ∈ [0.10, 0.30] is the peak additive line-integral magnitude
    # in μ_max-normalised units.  The literature baseline (Gjesteby 2016) is
    # ≈0.0075 for a 5 mm titanium implant at 100 keV; monoenergetic additive
    # models omit beam-hardening so a 10–50× amplification is standard to
    # produce visible, realistic streaks.
    #
    # p=0.3: metal implants are patient-specific and less common; lower
    # probability prevents the severe starvation streaks from dominating the
    # training distribution.
    "metal":              True,
    "metal_prob":         0.3,                    # per-sample Bernoulli probability
    "metal_n_objects":    (1, 2),                 # (min, max) number of implants
    "metal_scale":        (0.10, 0.30),           # peak additive normalised line-integral
    "metal_radius_frac":  (0.05, 0.35),           # implant radial position, fraction of image size
    "metal_library_path": None,                   # override; None → DEFAULT_METAL_LIBRARY
    "metal_starvation":   True,                   # enable photon-starvation clipping
}


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _motion_artifact(sino: np.ndarray, amplitude: float, n_cycles: float, phase: float) -> np.ndarray:
    """Simulate respiratory motion as a periodic lateral shift of detector data.

    Physical model
    --------------
    In parallel-beam CT a patient translation Δx at projection angle θ
    shifts the sinogram radially by  t → t − Δx·cosθ − Δy·sinθ
    (Lu & Mackie 2002, Phys. Med. Biol. 47:1267; Mao et al. 2007,
    Med. Phys. 34:3596).  For periodic breathing the displacement is:

        δt(θ) = amplitude · sin(2π · n_cycles · θ / N_angles + phase)

    Each projection row is shifted by its corresponding δt using
    linear interpolation (scipy.ndimage.shift, mode='nearest').
    After FBP this produces ghosting and edge-blurring consistent with
    clinical respiratory motion artifacts (Lu et al. 2005, Med. Phys.
    32:3295; DOI: 10.1118/1.2074187).

    Parameters
    ----------
    sino      : (N_angles, N_detectors) sinogram array.
    amplitude : Peak shift in detector bins (3–8 bins ≈ 2–6 mm).
    n_cycles  : Number of complete breathing cycles across the full scan.
    phase     : Initial phase offset in radians (randomised per sample).
    """
    N_angles = sino.shape[0]
    angle_idx = np.arange(N_angles, dtype=np.float32)
    displacement = amplitude * np.sin(2.0 * np.pi * n_cycles * angle_idx / N_angles + phase)

    result = np.empty_like(sino)
    for ai in range(N_angles):
        # shift along detector axis (axis=0 of the 1-D row)
        result[ai] = ndimage_shift(sino[ai], displacement[ai], mode='nearest')

    return result.astype(np.float32)


def _ring_artifact(
    sino:          np.ndarray,
    n_cols:        int,
    gain_range:    tuple,
    mu_max:        float = MU_MAX,
) -> np.ndarray:
    """Apply a detector-gain calibration error to random detector columns.

    Physical model
    --------------
    A detector element with multiplicative gain error g_c produces the observed
    photon count N_obs = g_c · N_true.  Pushing this through the LoDoPaB
    post-log normalisation (y = −ln(N/N0) / μ_max) gives a column-constant
    *additive* offset in the stored sinogram:

        y_obs(θ, t_c) = y_true(θ, t_c) + c_c,   c_c = −ln(g_c) / μ_max.

    Sampling strategy
    -----------------
    We draw the gain magnitude g from a realistic range gain_range = (g_lo, g_hi)
    with g_lo, g_hi < 1 (under-sensitive detectors → bright rings).  With
    probability 0.5 the element is instead modelled as over-sensitive by
    inverting g → 1/g, producing a negative offset (dark ring).  This keeps
    the perturbation symmetric in log-gain space and gives physically
    interpretable amplitudes:

        gain_range=(0.5, 0.95) → |c_c| ∈ [6.3e-4, 8.5e-3]    ≈ 5–50 % gain error
        gain_range=(0.3, 0.9)  → |c_c| ∈ [1.3e-3, 1.5e-2]    ≈ 10–70 % gain error

    After FBP the column-constant offset back-projects into a concentric ring
    centred on the axis of rotation (Sijbers & Postnov 2004, Prell et al. 2009).

    Parameters
    ----------
    gain_range : (g_lo, g_hi) with 0 < g_lo < g_hi < 1
        Under-sensitive gain range.  Over-sensitive gain (g > 1) is obtained
        by inverting with probability 0.5.
    mu_max : float
        μ_max normalisation constant (default: LoDoPaB's 81.35858 m⁻¹).
    """
    sino = sino.copy()
    cols = np.random.choice(NUM_DET_PIXELS, size=n_cols, replace=False)
    for col in cols:
        g = np.random.uniform(*gain_range)
        if np.random.rand() < 0.5:
            g = 1.0 / g                              # over-sensitive detector
        offset = -math.log(g) / mu_max               # c_c = −ln(g) / μ_max
        sino[:, col] += offset
    return sino


# ──────────────────────────────────────────────────────────────────────────────
# Metal artifact — library-based forward-projected implant
# ──────────────────────────────────────────────────────────────────────────────

_METAL_LIBRARY_CACHE: dict = {}


def _load_metal_library(path: Path) -> np.ndarray:
    """Load (and cache) the precomputed forward-projected metal-mask library.

    The library is produced by scripts/build_metal_library.py and contains
    per-sample-peak-normalised line-integral sinograms of centred metal masks.
    """
    key = str(path)
    if key not in _METAL_LIBRARY_CACHE:
        with np.load(path, allow_pickle=True) as npz:
            sinograms = npz["sinograms"].astype(np.float32)        # (N, A, D)
        _METAL_LIBRARY_CACHE[key] = sinograms
    return _METAL_LIBRARY_CACHE[key]


def _shift_sinogram_rows(sino: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Shift each projection row by an angle-dependent integer number of bins.

    Implements the Radon shift identity: a translation by (x0, y0) in the
    image domain produces a per-angle detector-axis shift
    δt(θ) = x0·cos θ + y0·sin θ.  Integer rounding is adequate here because
    the metal intensity is randomised and sub-pixel precision is masked by
    subsequent photon-starvation clipping.
    """
    out = np.zeros_like(sino)
    D   = sino.shape[1]
    for ai in range(sino.shape[0]):
        s = int(round(float(shifts[ai])))
        if   s == 0:   out[ai]      = sino[ai]
        elif s > 0:
            if s < D:  out[ai, s:]  = sino[ai, :D - s]
        else:
            k = -s
            if k < D:  out[ai, :D - k] = sino[ai, k:]
    return out


def _metal_artifact(
    sino:          np.ndarray,
    library:       np.ndarray,
    n_objects:     int,
    scale_range:   tuple,
    radius_range:  tuple,
    n0:            float,
    mu_max:        float,
    starvation:    bool = True,
) -> np.ndarray:
    """Add one or more metal implants to a post-log sinogram.

    Pipeline per implant
    --------------------
    1. Sample a centred metal sinogram s_0(θ, t) from the precomputed library.
    2. Sample an image-domain position (x0, y0) and apply the per-angle
       detector-axis shift δt(θ) = x0·cos θ + y0·sin θ (Radon shift identity).
    3. Scale the shifted sinogram by a per-sample peak magnitude drawn from
       scale_range and add it to the input sinogram.
    After all implants have been added, apply photon-starvation clipping:
    rays counting fewer than 1 photon at detector level are clipped to the
    noise floor, producing the characteristic metal-streak saturation.
    """
    sino   = sino.copy()
    angles = np.linspace(0.0, math.pi, NUM_ANGLES, endpoint=False)

    for _ in range(n_objects):
        idx     = int(np.random.randint(0, library.shape[0]))
        s0      = library[idx]                                      # (A, D)  peak ≈ 1

        r_lo, r_hi = radius_range
        r       = np.random.uniform(r_lo, r_hi) * IMAGE_SIZE        # px
        theta0  = np.random.uniform(0, 2.0 * math.pi)
        x0, y0  = r * math.cos(theta0), r * math.sin(theta0)
        shifts  = x0 * np.cos(angles) + y0 * np.sin(angles)         # (A,)

        scale   = np.random.uniform(*scale_range)
        sino   += scale * _shift_sinogram_rows(s0, shifts)

    if starvation:
        # Photon-starvation clipping in the detector (pre-log) domain.
        # Work in the un-normalised post-log variable y_phys = sino * μ_max;
        # detector count N = N0·exp(−y_phys) must be ≥ 1 (noise floor).
        y_phys = np.maximum(sino * mu_max, 0.0)
        N      = n0 * np.exp(-y_phys)
        N      = np.maximum(N, 1.0)
        sino   = (-np.log(N / n0) / mu_max).astype(np.float32)

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
    artifact_seed : int or None
        When set, artifact generation for each sample is seeded with
        ``artifact_seed + idx``, making every sample's artifact pattern
        deterministic and reproducible across epochs and runs.  Use this for
        validation and test splits so different models always see identical
        artifact realisations and comparisons are fair.
        When None (default), artifacts are drawn fresh each call (stochastic),
        which is the correct behaviour for training.
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
        artifact_seed:   int  = None,
        artifact_config: dict = None,
    ):
        if mode not in LEN:
            raise ValueError(f"mode must be one of {list(LEN)}, got '{mode}'")

        self.mode           = mode
        self.data_path      = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.add_artifacts  = add_artifacts
        self.artifact_seed  = artifact_seed
        self.geometry       = self.GEOMETRY.copy()
        self._length        = LEN[mode]

        # Merge user overrides into defaults
        self._artifact_cfg = {**DEFAULT_ARTIFACT_CONFIG, **(artifact_config or {})}

        # Lazily load the metal library if metal injection is enabled.
        # The library is shared across all Dataset instances via the
        # module-level _METAL_LIBRARY_CACHE dict (one entry per path).
        self._metal_library = None
        if self.add_artifacts and self._artifact_cfg.get("metal", False):
            lib_path = self._artifact_cfg.get("metal_library_path") or DEFAULT_METAL_LIBRARY
            lib_path = Path(lib_path)
            if not lib_path.exists():
                raise FileNotFoundError(
                    f"Metal library not found at {lib_path}.\n"
                    f"Run:  conda run -n tensor python scripts/build_metal_library.py"
                )
            self._metal_library = _load_metal_library(lib_path)

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
            a_lo, a_hi = cfg["motion_amplitude"]
            c_lo, c_hi = cfg["motion_cycles"]
            amplitude  = np.random.uniform(a_lo, a_hi)
            n_cycles   = np.random.uniform(c_lo, c_hi)
            phase      = np.random.uniform(0, 2 * np.pi)
            sino       = _motion_artifact(sino, amplitude, n_cycles, phase)

        if cfg.get("ring", False) and np.random.rand() < cfg.get("ring_prob", 1.0):
            n_lo, n_hi = cfg["ring_n_cols"]
            n_cols     = np.random.randint(n_lo, n_hi + 1)
            gain_range = cfg.get("ring_gain", (0.5, 0.95))
            sino       = _ring_artifact(sino, n_cols, gain_range)

        if (cfg.get("metal", False)
                and self._metal_library is not None
                and np.random.rand() < cfg.get("metal_prob", 1.0)):
            n_lo, n_hi = cfg["metal_n_objects"]
            n_obj      = np.random.randint(n_lo, n_hi + 1)
            sino       = _metal_artifact(
                sino,
                library      = self._metal_library,
                n_objects    = n_obj,
                scale_range  = cfg["metal_scale"],
                radius_range = cfg.get("metal_radius_frac", (0.05, 0.35)),
                n0           = N0,
                mu_max       = MU_MAX,
                starvation   = cfg.get("metal_starvation", True),
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
        #
        # When artifact_seed is set (validation / test), the numpy RNG state is
        # saved, seeded with (artifact_seed + idx) for this sample, then restored
        # afterwards.  This makes every sample's artifact pattern deterministic
        # and identical across epochs and different model runs, ensuring fair
        # comparison without polluting the global RNG used elsewhere.
        if self.add_artifacts:
            if self.artifact_seed is not None:
                rng_state = np.random.get_state()
                np.random.seed(self.artifact_seed + idx)

            sino = self._apply_artifacts(sino)
            sino = np.clip(sino, 0.0, 1.0)

            if self.artifact_seed is not None:
                np.random.set_state(rng_state)

        sino_t = torch.from_numpy(sino).unsqueeze(0)   # (1, 1000, 513)
        img_t  = torch.from_numpy(img ).unsqueeze(0)   # (1, 362, 362)

        return sino_t, img_t
