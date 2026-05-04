"""
test.py
=======
Evaluate a trained RED-CNN checkpoint on the LoDoPaB-CT artifact-injected test
set, reporting per-slice and aggregate PSNR / SSIM / RMSE.

Metric conventions match the rest of the baselines in this repo
(``baselines/bm3d_eval.py``, ``baselines/dm4ct_dps/test_dps.py``):

  * PSNR/SSIM use ``data_range = 1.0`` (LoDoPaB μ-normalised [0, 1]).
  * 20 visualisation indices come from a fixed ``VIS_SEED = 123`` so figures
    line up across baselines.
  * Display window for figures: [0, 0.45] (Leuschner et al. 2021, Fig. 5).

Usage
-----
    conda run -n tensor python baselines/redcnn/test.py \
        --ckpt checkpoints/baselines/redcnn/REDCNN_<iter>iter.ckpt
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extern" / "RED-CNN"))

from networks import RED_CNN                                   # noqa: E402

from trainer.parallel_fbp import DifferentiableParallelFBP  # noqa: E402
from utils.lodopab_dataset import LoDoPaBDataset                     # noqa: E402
from utils.metrics import compute_psnr, compute_ssim, compute_rmse   # noqa: E402

# ── Conventions shared with other baselines ───────────────────────────────────
N_SAVE_IMAGES      = 20
VIS_SEED           = 123
DATA_RANGE         = 1.0
TEST_ARTIFACT_SEED = 0
DISPLAY_VMIN, DISPLAY_VMAX = 0.0, 0.45

OUT_DIR = ROOT / "output" / "baselines" / "redcnn"
FIG_DIR = OUT_DIR / "fig"
LOG_DIR = OUT_DIR / "log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"test_{ts}.log"
    logger = logging.getLogger("redcnn_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path);  fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler();        sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def save_fig(noisy: np.ndarray, pred: np.ndarray, gt: np.ndarray,
             idx: int,
             noisy_m: tuple[float, float, float],
             pred_m:  tuple[float, float, float]):
    f, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(noisy, cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX)
    ax[0].set_title("FBP (noisy)")
    ax[0].set_xlabel(f"PSNR {noisy_m[0]:.2f} | SSIM {noisy_m[1]:.4f} | RMSE {noisy_m[2]:.4f}")
    ax[1].imshow(pred,  cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX)
    ax[1].set_title("RED-CNN")
    ax[1].set_xlabel(f"PSNR {pred_m[0]:.2f} | SSIM {pred_m[1]:.4f} | RMSE {pred_m[2]:.4f}")
    ax[2].imshow(gt,    cmap="gray", vmin=DISPLAY_VMIN, vmax=DISPLAY_VMAX)
    ax[2].set_title("Ground truth")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    f.tight_layout()
    f.savefig(FIG_DIR / f"sample_{idx:05d}.png", dpi=120, bbox_inches="tight")
    plt.close(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a REDCNN_<iter>iter.ckpt file")
    args = parser.parse_args()
    ckpt_path = Path(args.ckpt)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    logger.info(f"Checkpoint: {ckpt_path}")

    test_ds = LoDoPaBDataset("test", None,
                             add_artifacts=True, artifact_seed=TEST_ARTIFACT_SEED)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)
    logger.info(f"Test slices: {len(test_ds)}")

    fbp = DifferentiableParallelFBP.from_geometry(test_ds.geometry).to(DEVICE).eval()
    for p in fbp.parameters():
        p.requires_grad_(False)

    model = RED_CNN().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(seed=VIS_SEED)
    save_idx = set(rng.choice(len(test_ds),
                              size=min(N_SAVE_IMAGES, len(test_ds)),
                              replace=False).tolist())
    logger.info(f"Visualisation indices: {sorted(save_idx)}")

    sums = {"noisy_psnr": 0.0, "noisy_ssim": 0.0, "noisy_rmse": 0.0,
            "pred_psnr":  0.0, "pred_ssim":  0.0, "pred_rmse":  0.0}
    n = 0

    with torch.no_grad():
        for i, (sino, gt) in enumerate(tqdm(test_loader, desc="RED-CNN test")):
            sino = sino.to(DEVICE, non_blocking=True)
            gt   = gt  .to(DEVICE, non_blocking=True)

            noisy = fbp(sino).clamp(0.0, 1.0)
            pred  = model(noisy).clamp(0.0, 1.0)

            n_psnr = float(compute_psnr(noisy, gt, DATA_RANGE))
            n_ssim = float(compute_ssim(noisy, gt, DATA_RANGE))
            n_rmse = float(compute_rmse(noisy, gt))
            p_psnr = float(compute_psnr(pred,  gt, DATA_RANGE))
            p_ssim = float(compute_ssim(pred,  gt, DATA_RANGE))
            p_rmse = float(compute_rmse(pred,  gt))

            sums["noisy_psnr"] += n_psnr; sums["noisy_ssim"] += n_ssim; sums["noisy_rmse"] += n_rmse
            sums["pred_psnr"]  += p_psnr; sums["pred_ssim"]  += p_ssim; sums["pred_rmse"]  += p_rmse
            n += 1

            if i in save_idx:
                save_fig(
                    noisy.squeeze().cpu().numpy(),
                    pred .squeeze().cpu().numpy(),
                    gt   .squeeze().cpu().numpy(),
                    i,
                    (n_psnr, n_ssim, n_rmse),
                    (p_psnr, p_ssim, p_rmse),
                )

    metrics = {
        "n_slices": n,
        "checkpoint": str(ckpt_path),
        "noisy_fbp": {
            "psnr": sums["noisy_psnr"] / n,
            "ssim": sums["noisy_ssim"] / n,
            "rmse": sums["noisy_rmse"] / n,
        },
        "redcnn": {
            "psnr": sums["pred_psnr"] / n,
            "ssim": sums["pred_ssim"] / n,
            "rmse": sums["pred_rmse"] / n,
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"  FBP (noisy) | PSNR {metrics['noisy_fbp']['psnr']:.2f} "
                f"SSIM {metrics['noisy_fbp']['ssim']:.4f} "
                f"RMSE {metrics['noisy_fbp']['rmse']:.4f}")
    logger.info(f"  RED-CNN     | PSNR {metrics['redcnn']['psnr']:.2f} "
                f"SSIM {metrics['redcnn']['ssim']:.4f} "
                f"RMSE {metrics['redcnn']['rmse']:.4f}")
    logger.info(f"Wrote {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
