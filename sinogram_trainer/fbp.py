"""
sinogram_pipeline/fbp.py
========================

DifferentiableFBP
-----------------
PyTorch nn.Module wrapping diffCT's fan-beam FBP.  Used during training so
gradients flow back through reconstruction into the DnCNN weights.

radon_fbp
---------
Standalone reconstruction function using torch_radon.  Matches the exact
pipeline from helix2fan/reco_example_fan_beam.py and produces correctly
oriented CT images.  Use this for all visualisation and test metrics.

Key differences from a naive FBP:
  • Siemens flip: detector axis flipped before reconstruction.
  • Actual scan angles: taken from TIFF metadata + π/2 offset (not 0→2π).
  • Pixel-unit geometry: all distances divided by voxel_size; sinogram
    multiplied by 1/voxel_size before passing to the kernel.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from diffct.differentiable import (
    FanBackprojectorFunction,
    fan_cosine_weights,
    ramp_filter_1d,
)


class DifferentiableFBP(nn.Module):
    """Fan-beam FBP layer backed by diffCT CUDA kernels.

    Parameters
    ----------
    num_angles : int
        Number of projection angles (after any subsampling).
    num_detectors : int
        Number of detector elements per view (nu).
    sdd : float
        Source-to-Detector Distance in mm.
    sid : float
        Source-to-Isocenter Distance in mm.
    du : float
        Detector element spacing in mm (transverse).
    H, W : int
        Reconstruction image height / width in pixels.
    voxel_spacing : float
        Physical size of one reconstruction pixel in mm.
    detector_offset : float
        Lateral detector offset in mm (positive = shifted away from source).
        Computed as  (central_element_index − (nu−1)/2) × du.
    """

    def __init__(
        self,
        num_angles: int,
        num_detectors: int,
        sdd: float,
        sid: float,
        du: float,
        H: int,
        W: int,
        voxel_spacing: float = 0.7,
        detector_offset: float = 0.0,
    ):
        super().__init__()

        self.H = H
        self.W = W
        self.sdd = float(sdd)
        self.sid = float(sid)
        self.du = float(du)
        self.voxel_spacing = float(voxel_spacing)
        self.detector_offset = float(detector_offset)
        self.num_angles = num_angles
        # Angular integration weight (π / N_angles) for the discrete backprojection
        self.pi_over_N = math.pi / num_angles

        # Uniformly-spaced angles covering [0, 2π)
        angles = torch.arange(num_angles, dtype=torch.float32) * (
            2.0 * math.pi / num_angles
        )
        self.register_buffer("angles", angles)  # (A,)

        # Fan-beam cosine pre-weights: cos(γ) for each detector bin
        cos_w = fan_cosine_weights(
            num_detectors,
            du,
            sdd,
            detector_offset=detector_offset,
        )  # (D,)
        self.register_buffer("cos_w", cos_w)

    # ------------------------------------------------------------------

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of 2-D CT images via differentiable FBP.

        Parameters
        ----------
        sinogram : Tensor
            Shape (B, 1, num_angles, num_detectors) in physical
            log-attenuation units (NOT normalised).

        Returns
        -------
        Tensor
            Shape (B, 1, H, W)  –  reconstructed CT images.
        """
        # Accept (B, 1, A, D) → squeeze channel → (B, A, D)
        if sinogram.dim() == 4:
            sinogram = sinogram.squeeze(1)

        B = sinogram.shape[0]
        images = []

        for b in range(B):
            sino = sinogram[b]  # (A, D)

            # 1. Cosine pre-weighting (broadcast over angles)
            sino = sino * self.cos_w.unsqueeze(0)  # (A, D)

            # 2. Ramp filter along detector dimension
            sino = ramp_filter_1d(sino, dim=-1)  # (A, D)

            # 3. Weighted backprojection (differentiable — grads flow back to sino)
            img = FanBackprojectorFunction.apply(
                sino,
                self.angles,
                self.du,
                self.H,
                self.W,
                self.sdd,
                self.sid,
                self.voxel_spacing,
                self.detector_offset,
            )  # (H, W)

            # Angular integration weight + clip negative FBP artifacts
            img = torch.relu(img) * self.pi_over_N
            images.append(img)

        # Stack and add channel dimension → (B, 1, H, W)
        return torch.stack(images, dim=0).unsqueeze(1)

    @torch.no_grad()
    def reconstruct(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Same as forward but with no_grad (for target computation)."""
        return self.forward(sinogram)

    # ------------------------------------------------------------------
    # Convenience class method
    # ------------------------------------------------------------------

    @classmethod
    def from_geometry(cls, geo: dict, num_angles: int, H: int, W: int,
                      voxel_spacing: float = 0.7, angle_subsample: int = 1):
        """Construct from a geometry dict as stored in the TIFF metadata.

        Parameters
        ----------
        geo : dict
            Keys: nu, du, dso, dsd, rotview, det_central_element.
        num_angles : int
            Number of angles after subsampling (len of dataset items axis-0).
        H, W : int
            Reconstruction size.
        voxel_spacing : float
        angle_subsample : int
            Used only to verify num_angles == ceil(rotview / angle_subsample).
        """
        nu = int(geo.get("nu", 736))
        du = float(geo.get("du", 1.2858))
        sdd = float(geo.get("dsd", 1085.6))
        sid = float(geo.get("dso", 595.0))
        det_central = geo.get("det_central_element", [(nu - 1) / 2.0, 0.0])
        detector_offset = (float(det_central[0]) - (nu - 1) / 2.0) * du

        return cls(
            num_angles=num_angles,
            num_detectors=nu,
            sdd=sdd,
            sid=sid,
            du=du,
            H=H,
            W=W,
            voxel_spacing=voxel_spacing,
            detector_offset=detector_offset,
        )


# ---------------------------------------------------------------------------
# torch_radon-based FBP  (for visualisation / test metrics — no gradient)
# ---------------------------------------------------------------------------

def radon_fbp(
    sino_phys_np: np.ndarray,
    geometry: dict,
    angle_subsample: int = 1,
    recon_size: int = 512,
    voxel_size: float = 0.7,
    filter_name: str = "hann",
    device: str = "cuda",
) -> np.ndarray:
    """Fan-beam FBP via torch_radon — correctly oriented, no gradient.

    Follows helix2fan/reco_example_fan_beam.py exactly:
      1. Siemens flip  (flip detector axis)
      2. Scale sino by 1/voxel_size  (convert to pixel units)
      3. radon.filter_sinogram()  (cosine weight + chosen filter)
      4. radon.backprojection()

    Parameters
    ----------
    sino_phys_np  : np.ndarray  (num_angles, num_detectors)
                    Physical log-attenuation sinogram (already angle-subsampled).
    geometry      : dict
                    Keys: dso, ddo (or dsd), du, rotview, angles (list).
    angle_subsample : int
                    Factor used when building the sinogram (e.g. 4 → 576 views
                    from 2304).  Used to index into the metadata angle list.
    recon_size    : int   Output H = W in pixels.
    voxel_size    : float mm per pixel (0.7 for 512×512 Mayo LDCT).
    filter_name   : str   'hann' | 'ram-lak' | 'shepp-logan' | 'cosine' | 'hamming'
    device        : str   'cuda' or 'cpu'.

    Returns
    -------
    reco_hu : np.ndarray  (recon_size, recon_size)  in Hounsfield Units
    """
    from torch_radon import RadonFanbeam

    # 1. Siemens flip
    prj = np.flip(sino_phys_np, axis=1).copy()   # (A, D)
    num_angles = prj.shape[0]

    # 2. Angles from metadata, subsampled + π/2 offset
    meta_angles = geometry.get("angles", None)
    if meta_angles is not None:
        angles = np.array(meta_angles)[::angle_subsample][:num_angles] + (np.pi / 2)
    else:
        # Fallback: uniform spacing (less accurate)
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False) + (np.pi / 2)

    # 3. Geometry in pixel units
    vox_scaling = 1.0 / voxel_size
    dso = float(geometry.get("dso", 595.0))
    dsd = float(geometry.get("dsd", 1085.6))
    ddo = float(geometry.get("ddo", dsd - dso))   # detector-to-isocenter
    du  = float(geometry.get("du",  1.2858))

    radon = RadonFanbeam(
        recon_size,
        angles,
        source_distance=vox_scaling * dso,
        det_distance=vox_scaling * ddo,
        det_count=prj.shape[1],
        det_spacing=vox_scaling * du,
        clip_to_circle=False,
    )

    # 4. Scale sinogram, filter, backproject
    dev    = torch.device(device)
    sino_t = torch.tensor(prj * vox_scaling, dtype=torch.float32).to(dev)

    with torch.no_grad():
        filtered = radon.filter_sinogram(sino_t, filter_name=filter_name)
        reco     = radon.backprojection(filtered)

    reco_np = reco.cpu().numpy()

    # 5. Convert to Hounsfield Units  (matches helix2fan reco_example_fan_beam.py)
    #    HU = 1000 * (mu - mu_water) / mu_water
    hu_factor = float(geometry.get("hu_factor", 0.0192))
    return 1000.0 * (reco_np - hu_factor) / hu_factor
