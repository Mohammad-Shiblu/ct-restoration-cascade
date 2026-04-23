"""
baselines/bm3d_eval.py
======================
Evaluate BM3D as a classical baseline on the LoDoPaB artifact test set.

Pipeline
--------
    noisy_sino (artifact-injected) → FBP → noisy_img → BM3D → denoised_img

Sigma tuning
------------
    BM3D requires a noise std estimate (sigma_psd).  This script first tunes
    sigma on a small validation subset by grid search (maximise PSNR), then
    evaluates with the best sigma on the full test set.

Results saved to
----------------
    output/baselines/bm3d/
        metrics.json          — aggregate PSNR / SSIM / RMSE
        sigma_search.json     — per-sigma val PSNR for the grid search
        images/sample_<idx>.png — 20 fixed visual comparisons (seed=123)

Usage
-----
    conda run -n tensor python baselines/bm3d_eval.py
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import bm3d

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sinogram_trainer.parallel_fbp import DifferentiableParallelFBP
from utils.lodopab_dataset import LoDoPaBDataset
from utils.metrics import compute_psnr, compute_ssim, compute_rmse

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR         = "output/baselines/bm3d"
VAL_TUNE_SAMPLES   = 200
SIGMA_GRID         = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
N_SAVE_IMAGES      = 20
# Fixed seed — identical save_idx across ALL baseline/model evaluations
VIS_SEED           = 123
DATA_RANGE         = 1.0
VAL_ARTIFACT_SEED  = 42
TEST_ARTIFACT_SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def sino_to_img(fbp, noisy_sino: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return fbp(noisy_sino)


def apply_bm3d(noisy_np: np.ndarray, sigma: float) -> np.ndarray:
    out = bm3d.bm3d(noisy_np.astype(np.float64), sigma_psd=sigma,
                    stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.cpu().squeeze().numpy()


def make_save_idx(n_test: int) -> set:
    """Fixed indices for visual comparison — same seed used by all models."""
    rng = np.random.default_rng(seed=VIS_SEED)
    return set(rng.choice(n_test, size=min(N_SAVE_IMAGES, n_test),
                           replace=False).tolist())


# ── Sigma grid search on validation subset ────────────────────────────────────

def tune_sigma(fbp, val_loader) -> tuple[float, dict]:
    print(f"\n[BM3D] Sigma grid search on {VAL_TUNE_SAMPLES} validation samples …")
    print(f"       Grid: {SIGMA_GRID}\n")

    sigma_psnr = {s: 0.0 for s in SIGMA_GRID}
    count = 0

    for noisy_sino, gt_img in val_loader:
        if count >= VAL_TUNE_SAMPLES:
            break
        noisy_sino = noisy_sino.to(DEVICE)
        noisy_img  = sino_to_img(fbp, noisy_sino)

        noisy_np = tensor_to_np(noisy_img)
        gt_np    = tensor_to_np(gt_img)

        for sigma in SIGMA_GRID:
            denoised = apply_bm3d(noisy_np, sigma)
            psnr = compute_psnr(
                torch.tensor(denoised).unsqueeze(0).unsqueeze(0),
                torch.tensor(gt_np).unsqueeze(0).unsqueeze(0),
                data_range=DATA_RANGE,
            )
            sigma_psnr[sigma] += psnr
        count += 1

        if count % 50 == 0:
            print(f"  Tuned {count}/{VAL_TUNE_SAMPLES} …")

    avg = {s: v / count for s, v in sigma_psnr.items()}
    best_sigma = max(avg, key=avg.get)

    print("\n  Sigma search results (avg PSNR on val subset):")
    for s, p in sorted(avg.items()):
        marker = " ← best" if s == best_sigma else ""
        print(f"    sigma={s:.2f}  PSNR={p:.2f} dB{marker}")

    return best_sigma, avg


# ── Full test evaluation ──────────────────────────────────────────────────────

def evaluate(fbp, test_loader, sigma: float, save_dir: str) -> dict:
    print(f"\n[BM3D] Evaluating on full test set (sigma={sigma:.2f}) …")

    total_psnr_noisy = total_ssim_noisy = total_rmse_noisy = 0.0
    total_psnr_bm3d  = total_ssim_bm3d  = total_rmse_bm3d  = 0.0
    n = 0

    n_test   = len(test_loader.dataset)
    save_idx = make_save_idx(n_test)
    print(f"  Saving visualisations for indices: {sorted(save_idx)}")

    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    t0 = time.time()
    for i, (noisy_sino, gt_img) in enumerate(
        tqdm(test_loader, desc="BM3D test eval")
    ):
        noisy_sino = noisy_sino.to(DEVICE)
        noisy_img  = sino_to_img(fbp, noisy_sino)

        noisy_np   = tensor_to_np(noisy_img)
        gt_np      = tensor_to_np(gt_img)
        denoised   = apply_bm3d(noisy_np, sigma)

        noisy_t    = torch.tensor(noisy_np).unsqueeze(0).unsqueeze(0)
        gt_t       = torch.tensor(gt_np).unsqueeze(0).unsqueeze(0)
        denoised_t = torch.tensor(denoised).unsqueeze(0).unsqueeze(0)

        total_psnr_noisy += compute_psnr(noisy_t,    gt_t, DATA_RANGE)
        total_ssim_noisy += compute_ssim(noisy_t,    gt_t, DATA_RANGE)
        total_rmse_noisy += float(compute_rmse(noisy_t, gt_t))

        total_psnr_bm3d  += compute_psnr(denoised_t, gt_t, DATA_RANGE)
        total_ssim_bm3d  += compute_ssim(denoised_t, gt_t, DATA_RANGE)
        total_rmse_bm3d  += float(compute_rmse(denoised_t, gt_t))

        n += 1

        if i in save_idx:
            _save_comparison(i, noisy_np, denoised, gt_np,
                             noisy_t, denoised_t, gt_t, img_dir, sigma)

    elapsed = time.time() - t0
    print(f"  Done. Elapsed: {elapsed/60:.1f} min  ({elapsed/n:.1f} s/image)")

    return {
        "n_test":  n,
        "sigma":   sigma,
        "noisy_fbp": {
            "psnr": total_psnr_noisy / n,
            "ssim": total_ssim_noisy / n,
            "rmse": total_rmse_noisy / n,
        },
        "bm3d": {
            "psnr": total_psnr_bm3d / n,
            "ssim": total_ssim_bm3d / n,
            "rmse": total_rmse_bm3d / n,
        },
    }


def _save_comparison(idx, noisy_np, denoised_np, gt_np,
                     noisy_t, denoised_t, gt_t, img_dir, sigma):
    vmin, vmax = 0.0, 0.45
    DR = DATA_RANGE

    psnr_n = compute_psnr(noisy_t,    gt_t, DR)
    ssim_n = compute_ssim(noisy_t,    gt_t, DR)
    psnr_b = compute_psnr(denoised_t, gt_t, DR)
    ssim_b = compute_ssim(denoised_t, gt_t, DR)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, arr, title in zip(
        axes,
        [noisy_np, denoised_np, gt_np],
        [
            f"FBP (noisy)\nPSNR={psnr_n:.2f}  SSIM={ssim_n:.4f}",
            f"BM3D (σ={sigma:.2f})\nPSNR={psnr_b:.2f}  SSIM={ssim_b:.4f}",
            "Ground Truth",
        ],
    ):
        ax.imshow(np.flipud(arr.T), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"sample_{idx:04d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[BM3D] Loading datasets …")
    val_ds  = LoDoPaBDataset("validation", add_artifacts=True,
                             artifact_seed=VAL_ARTIFACT_SEED)
    test_ds = LoDoPaBDataset("test",       add_artifacts=True,
                             artifact_seed=TEST_ARTIFACT_SEED)

    val_loader  = DataLoader(val_ds,  batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True)

    fbp = DifferentiableParallelFBP.from_geometry(val_ds.geometry).to(DEVICE)

    # 1. Sigma tuning on validation subset
    best_sigma, sigma_search = tune_sigma(fbp, val_loader)
    with open(os.path.join(OUTPUT_DIR, "sigma_search.json"), "w") as f:
        json.dump({str(k): v for k, v in sigma_search.items()}, f, indent=2)

    # 2. Full test evaluation
    results = evaluate(fbp, test_loader, best_sigma, OUTPUT_DIR)

    # 3. Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # 4. Print summary
    print("\n" + "=" * 60)
    print("BM3D BASELINE RESULTS")
    print("=" * 60)
    nb = results["noisy_fbp"]
    bm = results["bm3d"]
    print(f"  Noisy FBP : PSNR={nb['psnr']:.2f}  SSIM={nb['ssim']:.4f}  RMSE={nb['rmse']:.4f}")
    print(f"  BM3D      : PSNR={bm['psnr']:.2f}  SSIM={bm['ssim']:.4f}  RMSE={bm['rmse']:.4f}")
    print(f"  Gain      : ΔPSNR={bm['psnr']-nb['psnr']:+.2f}  ΔSSIM={bm['ssim']-nb['ssim']:+.4f}")
    print(f"  Best sigma: {best_sigma}")
    print(f"  Results   : {metrics_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
