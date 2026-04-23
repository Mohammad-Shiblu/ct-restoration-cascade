"""Pre-compute a library of forward-projected metal-implant masks.

Rationale
---------
For the physics-informed metal-artifact model (Chapter 3 §3.5) we insert a
dense implant into the image domain, forward-project it through the LoDoPaB
parallel-beam geometry to obtain its line-integral contribution
s_metal(θ, t), and add that contribution to the pre-noised LoDoPaB sinogram.
Forward projection of a 362×362 image at 1000 angles is too slow to perform
inside the DataLoader workers (no CUDA in worker processes, CPU radon is
~0.5 s/sample).  We therefore forward-project a fixed library of centred
metal masks once offline (this script) and, at training time, recover the
sinogram contribution of an implant at arbitrary position (x0, y0) by
applying a per-angle detector-axis shift

    δt(θ) = x0·cos θ + y0·sin θ

(Radon shift identity — see methodology.tex eq. corr_motion_shift).

Output
------
    data/metal/metal_library.npz
        masks     : (N, 362, 362)  float16  — image-domain metal masks
        sinograms : (N, 1000, 513) float16  — forward-projected sinograms
        meta      : dict with geometry and generation parameters

Usage
-----
    conda run -n tensor python scripts/build_metal_library.py --n 200
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch_radon import Radon

REPO = Path(__file__).resolve().parents[1]

# LoDoPaB parallel-beam geometry (must match utils/lodopab_dataset.py)
NUM_ANGLES       = 1000
NUM_DET_PIXELS   = 513
IMAGE_SIZE       = 362
DETECTOR_SPACING = IMAGE_SIZE * math.sqrt(2) / NUM_DET_PIXELS


def _disk(h: int, w: int, r: float) -> np.ndarray:
    """Binary disk of radius r (pixels) centred at image centre."""
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).astype(np.float32)


def _ellipse(h: int, w: int, a: float, b: float, phi: float) -> np.ndarray:
    """Filled ellipse with semi-axes (a, b) rotated by phi (radians)."""
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = yy - cy, xx - cx
    xr =  x * math.cos(phi) + y * math.sin(phi)
    yr = -x * math.sin(phi) + y * math.cos(phi)
    return ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0).astype(np.float32)


def _rectangle(h: int, w: int, dx: float, dy: float, phi: float) -> np.ndarray:
    """Filled rectangle of extent (2·dx, 2·dy) rotated by phi (radians)."""
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = yy - cy, xx - cx
    xr =  x * math.cos(phi) + y * math.sin(phi)
    yr = -x * math.sin(phi) + y * math.cos(phi)
    return ((np.abs(xr) <= dx) & (np.abs(yr) <= dy)).astype(np.float32)


def _l_shape(h: int, w: int, length: float, thickness: float, phi: float) -> np.ndarray:
    """L-shaped bracket (e.g. orthopaedic plate)."""
    arm1 = _rectangle(h, w, length,    thickness, phi)
    arm2 = _rectangle(h, w, thickness, length,    phi)
    return np.clip(arm1 + arm2, 0.0, 1.0)


def generate_library(
    n: int,
    size_range_px: tuple[float, float] = (3.0, 12.0),
    seed: int = 0,
) -> np.ndarray:
    """Generate N centred metal masks covering a variety of shapes and sizes.

    Size range 3–12 px with LoDoPaB pitch ≈0.72 mm → cross-sections 2–9 mm,
    consistent with dental fillings, surgical clips and small orthopaedic
    hardware.  All masks are centred at image centre; runtime translates
    them via the Radon shift identity (no re-projection needed).
    """
    rng   = np.random.default_rng(seed)
    masks = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    kinds = rng.choice(["disk", "ellipse", "rect", "lshape"],
                        size=n, p=[0.35, 0.35, 0.15, 0.15])

    for i, k in enumerate(kinds):
        s_lo, s_hi = size_range_px
        if k == "disk":
            masks[i] = _disk(IMAGE_SIZE, IMAGE_SIZE, rng.uniform(s_lo, s_hi))
        elif k == "ellipse":
            a = rng.uniform(s_lo, s_hi)
            b = rng.uniform(s_lo * 0.5, a)         # flatter than long axis
            masks[i] = _ellipse(IMAGE_SIZE, IMAGE_SIZE,
                                a, b, rng.uniform(0, math.pi))
        elif k == "rect":
            dx = rng.uniform(s_lo * 0.5, s_hi * 0.6)
            dy = rng.uniform(s_lo * 0.5, s_hi * 0.6)
            masks[i] = _rectangle(IMAGE_SIZE, IMAGE_SIZE,
                                  dx, dy, rng.uniform(0, math.pi))
        else:
            masks[i] = _l_shape(IMAGE_SIZE, IMAGE_SIZE,
                                rng.uniform(s_lo * 0.8, s_hi),
                                rng.uniform(1.5, 3.0),
                                rng.uniform(0, math.pi))
    return masks


def forward_project(masks: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Forward-project a stack of image-domain masks via torch_radon.

    Returns the line-integral sinograms normalised to peak 1.0 per sample,
    so that the runtime `metal_scale` parameter directly controls the
    additive contribution magnitude (in μ_max-normalised units).
    """
    angles = np.linspace(0.0, math.pi, NUM_ANGLES, endpoint=False).astype(np.float32)
    radon  = Radon(IMAGE_SIZE, angles,
                    det_count=NUM_DET_PIXELS,
                    det_spacing=DETECTOR_SPACING)

    x       = torch.from_numpy(masks).to(device)                   # (N, H, W)
    sinos_t = radon.forward(x)                                      # (N, A, D)
    sinos   = sinos_t.cpu().numpy()

    # Normalise each sinogram to peak 1.0.  This decouples the library's
    # arbitrary mask-intensity scale from the runtime physical magnitude,
    # which is set by `metal_scale ∈ [0.1, 0.2]` in the dataset config.
    peaks = sinos.reshape(sinos.shape[0], -1).max(axis=1)
    peaks[peaks < 1e-6] = 1.0
    sinos = sinos / peaks[:, None, None]
    return sinos.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=200,
                    help="Number of metal masks to generate (default: 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",  type=str,
                    default=str(REPO / "data" / "metal" / "metal_library.npz"))
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[build_metal_library] generating N={args.n} masks on {args.device} …")
    masks = generate_library(args.n, seed=args.seed)
    print(f"[build_metal_library] forward-projecting (this needs a GPU) …")
    sinos = forward_project(masks, device=args.device)

    meta = {
        "num_angles":       NUM_ANGLES,
        "num_detectors":    NUM_DET_PIXELS,
        "image_size":       IMAGE_SIZE,
        "detector_spacing": DETECTOR_SPACING,
        "n":                args.n,
        "seed":             args.seed,
        "normalisation":    "per-sample peak = 1.0",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        masks     = masks.astype(np.float16),
        sinograms = sinos.astype(np.float16),
        meta      = np.array([meta], dtype=object),
    )
    size_mb = out.stat().st_size / (1024 ** 2)
    print(f"[build_metal_library] saved {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
