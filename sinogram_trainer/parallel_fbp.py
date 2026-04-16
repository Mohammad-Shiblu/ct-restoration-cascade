"""
parallel_fbp.py
===============
Differentiable parallel-beam FBP for LoDoPaB-CT using torch_radon CUDA kernels.

torch_radon processes the full batch in a single CUDA call (~48x faster than
the previous diffCT sample-by-sample loop) and supports gradients through both
FBP and forward projection.

Usage
-----
    fbp = DifferentiableParallelFBP.from_lodopab().to(device)
    images = fbp(sinograms)          # (B, 1, 1000, 513) → (B, 1, 362, 362)
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch_radon import Radon


class DifferentiableParallelFBP(nn.Module):
    """Differentiable parallel-beam FBP layer backed by torch_radon CUDA kernels.

    Parameters
    ----------
    num_angles : int
        Number of projection angles.
    num_detectors : int
        Number of detector elements per view.
    H, W : int
        Reconstruction image height / width in pixels.  Must be equal
        (torch_radon requires a square reconstruction grid).
    detector_spacing : float
        Physical spacing between detector elements in the same units as
        voxel_spacing.  For LoDoPaB in pixel units: IMAGE_SIZE*sqrt(2) / NUM_DET
        ≈ 0.996.
    voxel_spacing : float
        Unused — kept for API compatibility with from_geometry().
    angle_min, angle_max : float
        First and last projection angle in radians.  Default: 0 and π.
    """

    def __init__(
        self,
        num_angles:       int   = 1000,
        num_detectors:    int   = 513,
        H:                int   = 362,
        W:                int   = 362,
        detector_spacing: float = 1.0,
        voxel_spacing:    float = 1.0,
        angle_min:        float = 0.0,
        angle_max:        float = math.pi,
    ):
        super().__init__()

        self.H = H
        self.W = W
        self._num_angles = num_angles

        angles = np.linspace(angle_min, angle_max, num_angles, endpoint=False).astype(np.float32)
        self.radon = Radon(H, angles, det_count=num_detectors, det_spacing=detector_spacing)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of CT images via differentiable parallel-beam FBP.

        Parameters
        ----------
        sinogram : Tensor
            Shape (B, 1, num_angles, num_detectors).

        Returns
        -------
        Tensor
            Shape (B, 1, H, W) — reconstructed CT images.
        """
        s = sinogram.squeeze(1)                                    # (B, A, D)
        filtered = self.radon.filter_sinogram(s)                 # (B, A, D)
        # torch_radon backprojection averages over angles internally,
        # but filter_sinogram already applied π/(2N) expecting a sum —
        # multiply by N_angles to recover the correct FBP scaling.
        images   = self.radon.backprojection(filtered) * self._num_angles  # (B, H, W)
        # torch_radon uses y-upward convention → flip vertically to match
        # standard image (y-downward) orientation used by the GT and trainer.
        images = torch.rot90(images, k=-1, dims=[-2, -1])
        return images.unsqueeze(1)                               # (B, 1, H, W)

    @torch.no_grad()
    def reconstruct(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Same as forward but with no_grad (for visualisation / metrics)."""
        return self.forward(sinogram)

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_lodopab(cls, **kwargs) -> "DifferentiableParallelFBP":
        """Construct with the exact LoDoPaB-CT geometry."""
        defaults = dict(
            num_angles       = 1000,
            num_detectors    = 513,
            H                = 362,
            W                = 362,
            detector_spacing = 362.0 * math.sqrt(2) / 513.0,
            voxel_spacing    = 1.0,
            angle_min        = 0.0,
            angle_max        = math.pi,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def from_geometry(cls, geometry: dict, **kwargs) -> "DifferentiableParallelFBP":
        """Construct from a geometry dict (as stored in LoDoPaBDataset.GEOMETRY)."""
        defaults = dict(
            num_angles       = geometry["num_angles"],
            num_detectors    = geometry["num_detectors"],
            H                = geometry["image_size"],
            W                = geometry["image_size"],
            detector_spacing = geometry["detector_spacing"],
            voxel_spacing    = geometry["voxel_spacing"],
        )
        defaults.update(kwargs)
        return cls(**defaults)
