import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.unet    import UNet
from models.resunet import ResUNet
from trainer.parallel_fbp import DifferentiableParallelFBP
from utils.lodopab_dataset import LoDoPaBDataset
from utils.metrics import compute_psnr, compute_ssim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT   = Path(__file__).resolve().parent.parent   # .../Masters-Thesis/

def load_model(config_path, ckpt_dir_override=None):
    """Build stages + FBP from a deep-supervision config and load best checkpoints.

    Returns
    -------
    stages : nn.ModuleList  (eval mode, on DEVICE)
    fbp    : DifferentiableParallelFBP  (eval mode, on DEVICE, no grad)
    config : dict
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)

    num_stages  = config["num_stages"]
    stage1_mode = config.get("stage1_mode", "residual")
    dual_domain = config.get("dual_domain", False)

    stages = nn.ModuleList()
    for s in range(num_stages):
        if s == 0:
            features = config.get("stage0_features", [32, 64, 128, 256])
            model = UNet(
                in_channels=1, out_channels=1,
                features=features, output_activation="sigmoid",
            )
        elif stage1_mode == "direct":
            features = config.get(
                "stage1_features",
                config.get("stage0_features", [32, 64, 128, 256]),
            )
            model = UNet(
                in_channels=1, out_channels=1,
                features=features, output_activation="sigmoid",
            )
        else:  # residual cascade
            features = config.get("stage1_features", [32, 64, 128, 256])
            in_ch    = 3 if dual_domain else 1
            model = ResUNet(
                in_channels=in_ch, out_channels=1,
                features=features, output_activation="tanh",
            )
        stages.append(model)

    if ckpt_dir_override is not None:
        ckpt_dir = Path(ckpt_dir_override)
    else:
        ckpt_dir = (
            ROOT 
            / config.get("checkpoints_dir", "checkpoints")
            / config["model"]
            / config["test_no"]
        )

    for s, m in enumerate(stages):
        ckpt_path = ckpt_dir / f"stage_{s}_best.pth"
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m.to(DEVICE).eval()
        n = sum(p.numel() for p in m.parameters())
        print(f"Stage {s}: {n:,} params  ({ckpt_path.name})")

    geom_ds = LoDoPaBDataset("test", None)
    fbp = DifferentiableParallelFBP.from_geometry(geom_ds.geometry).to(DEVICE).eval()
    for p in fbp.parameters():
        p.requires_grad_(False)

    arch_tag = (
        f"{num_stages}-stage "
        + ("naive cascade" if stage1_mode == "direct" else "residual cascade")
        + (" [dual-domain]" if dual_domain else "")
        if num_stages > 1
        else "single-stage U-Net"
    )
    print(f"Loaded: {config['test_no']}  ({arch_tag})")
    return stages, fbp, config

def load_redcnn(ckpt_path=None):
    """Load a trained RED-CNN from checkpoint.

    Returns
    -------
    model : RED_CNN  (eval mode, on DEVICE)
    fbp   : DifferentiableParallelFBP  (eval mode, on DEVICE, no grad)
    """
    import sys as _sys
    redcnn_dir = ROOT / "extern" / "RED-CNN"
    if str(redcnn_dir) not in _sys.path:
        _sys.path.insert(0, str(redcnn_dir))
    from networks import RED_CNN  # noqa: E402

    if ckpt_path is None:
        ckpt_path = ROOT / "checkpoints" / "baselines" / "redcnn" / "REDCNN_best.ckpt"
    ckpt_path = Path(ckpt_path)

    model = RED_CNN().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"RED-CNN: {n:,} params  ({ckpt_path.name})")

    geom_ds = LoDoPaBDataset("test", None)
    fbp = DifferentiableParallelFBP.from_geometry(geom_ds.geometry).to(DEVICE).eval()
    for p in fbp.parameters():
        p.requires_grad_(False)

    return model, fbp


@torch.no_grad()
def run_redcnn(model, fbp, sino):
    """Run FBP → RED-CNN. Returns CPU tensors (1,1,H,W).

    Returns
    -------
    noisy_t : CPU tensor (1,1,H,W)
    pred_t  : CPU tensor (1,1,H,W)
    """
    sino    = sino.to(DEVICE)
    noisy_t = fbp(sino).clamp(0.0, 1.0)
    pred_t  = model(noisy_t).clamp(0.0, 1.0)
    return noisy_t.cpu(), pred_t.cpu()


@torch.no_grad()
def run_bm3d(fbp, sino, sigma=0.1):
    """Run FBP → BM3D. Returns CPU tensors (1,1,H,W).

    Returns
    -------
    noisy_t    : CPU tensor (1,1,H,W)
    denoised_t : CPU tensor (1,1,H,W)
    """
    import bm3d as _bm3d
    sino     = sino.to(DEVICE)
    noisy_t  = fbp(sino).clamp(0.0, 1.0).cpu()
    noisy_np = noisy_t.squeeze().numpy()
    denoised = _bm3d.bm3d(
        noisy_np.astype(np.float64), sigma_psd=sigma,
        stage_arg=_bm3d.BM3DStages.ALL_STAGES,
    )
    denoised_t = torch.from_numpy(
        np.clip(denoised, 0.0, 1.0).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)
    return noisy_t, denoised_t


def plot_baseline(noisy_t, pred_t, gt_t, method_label,
                  slice_idx=None, vmin=0.0, vmax=0.45):
    """Plot FBP | <method> | Ground Truth with PSNR/SSIM.

    All inputs are CPU tensors (1,1,H,W).
    """
    DR = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _show(ax, t, title):
        ax.imshow(_to_display(t), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fbp_psnr = compute_psnr(noisy_t, gt_t, data_range=DR)
    fbp_ssim = compute_ssim(noisy_t, gt_t, data_range=DR)
    _show(axes[0], noisy_t, f"FBP (noisy)\nPSNR {fbp_psnr:.2f} dB  SSIM {fbp_ssim:.4f}")

    pred_psnr = compute_psnr(pred_t, gt_t, data_range=DR)
    pred_ssim = compute_ssim(pred_t, gt_t, data_range=DR)
    _show(axes[1], pred_t, f"{method_label}\nPSNR {pred_psnr:.2f} dB  SSIM {pred_ssim:.4f}")

    _show(axes[2], gt_t, "Ground Truth")

    idx_str = f"  |  slice {slice_idx}" if slice_idx is not None else ""
    fig.suptitle(f"{method_label}{idx_str}", fontsize=10)
    plt.tight_layout()
    return fig


@torch.no_grad()
def run_inference(stages, fbp, sino, config):
    """Run FBP → cascade. Returns CPU tensors (1,1,H,W) — no display transform.

    Returns
    -------
    noisy_t       : CPU tensor (1,1,H,W)
    stage_tensors : list of CPU tensors (1,1,H,W), one per stage
    """
    sino = sino.to(DEVICE)
    num_stages  = config["num_stages"]
    stage1_mode = config.get("stage1_mode", "residual")
    dual_domain = config.get("dual_domain", False)

    noisy_t = fbp(sino).clamp(0.0, 1.0)

    hann_t = sirt_t = None
    if dual_domain and num_stages > 1:
        hann_t = fbp.hann_fbp(sino)
        sirt_t = fbp.shepp_logan_fbp(sino)

    x = stages[0](noisy_t)
    stage_tensors = [x]

    for s in range(1, num_stages):
        if stage1_mode == "direct":
            x = stages[s](x)
        else:
            inp = torch.cat([x, hann_t, sirt_t], dim=1) if hann_t is not None else x
            x   = torch.clamp(x + stages[s](inp), 0.0, 1.0)
        stage_tensors.append(x)

    return noisy_t.cpu(), [t.cpu() for t in stage_tensors]

def _to_display(t):
    """Tensor (1,1,H,W) → display-oriented (H,W) numpy array (flipud + T).
    Applied only for imshow — metrics are always computed from tensors.
    """
    return np.flipud(t.squeeze().numpy().T)

def plot_results(noisy_t, stage_tensors, gt_t, config,
                 slice_idx=None, vmin=0.0, vmax=0.45):
    """Plot FBP | Stage 0 | … | Stage N-1 | Ground Truth with PSNR/SSIM.

    All inputs are CPU tensors (1,1,H,W).
    PSNR/SSIM are computed from tensors before any display transform.

    Parameters
    ----------
    noisy_t       : CPU tensor (1,1,H,W) — FBP output from run_inference.
    stage_tensors : list of CPU tensors (1,1,H,W) from run_inference.
    gt_t          : CPU tensor (1,1,H,W) — ground truth.
    config        : config dict returned by load_model.
    slice_idx     : optional test slice index shown in the suptitle.
    vmin, vmax    : display window (default [0, 0.45], Leuschner et al. 2021).
    """
    DR = 1.0

    n_panels = 2 + len(stage_tensors)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    def _show(ax, t, title):
        ax.imshow(_to_display(t), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Metrics computed from tensors (no stride issues from flipud)
    fbp_psnr = compute_psnr(noisy_t, gt_t, data_range=DR)
    fbp_ssim = compute_ssim(noisy_t, gt_t, data_range=DR)
    _show(axes[0], noisy_t, f"FBP (noisy)\nPSNR {fbp_psnr:.2f} dB  SSIM {fbp_ssim:.4f}")

    for s, out_t in enumerate(stage_tensors):
        s_psnr = compute_psnr(out_t, gt_t, data_range=DR)
        s_ssim = compute_ssim(out_t, gt_t, data_range=DR)
        label  = "Single-stage" if len(stage_tensors) == 1 else f"Stage {s+1}"
        _show(axes[s + 1], out_t, f"{label}\nPSNR {s_psnr:.2f} dB  SSIM {s_ssim:.4f}")

    _show(axes[-1], gt_t, "Ground Truth")

    idx_str  = f"  |  slice {slice_idx}" if slice_idx is not None else ""
    stage1   = config.get("stage1_mode", "residual") if config["num_stages"] > 1 else ""
    dual_tag = "  [dual-domain]" if config.get("dual_domain") else ""
    fig.suptitle(
        f"{config['test_no']}{idx_str}\n"
        f"{config['num_stages']}-stage  {stage1}{dual_tag}",
        fontsize=10,
    )
    plt.tight_layout()
    return fig

