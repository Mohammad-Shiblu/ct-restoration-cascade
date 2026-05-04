"""
train.py
========
Train RED-CNN (Chen et al. 2017, IEEE TMI) on LoDoPaB-CT.

The model definition is imported verbatim from the upstream repo vendored under
``extern/RED-CNN/networks.py`` (commit at clone time).  The training procedure
(MSE loss, Adam, lr=1e-5, lr×0.5 every 3000 iters, 64×64 random patches with
patch_n=10, batch_size=16, 100 epochs) is replicated from the upstream
``solver.py``.  Only the data pipeline is adapted to LoDoPaB:

  * upstream reads ``*_input.npy`` / ``*_target.npy`` AAPM pairs from disk;
  * here, sinograms come from ``LoDoPaBDataset(add_artifacts=True)`` and are
    FBP'd on-the-fly through ``DifferentiableParallelFBP`` (parallel-beam
    1000×513 → 362²).  The (FBP_artifact, GT_clean) pair stays in LoDoPaB's
    μ-normalised [0, 1] space — the same input distribution every other
    baseline in this thesis trains on.

Citation chain (RED-CNN as published, trained on LoDoPaB artifact-injected
inputs vs clean GT) holds — only the dataset and FBP step are LoDoPaB-specific.

References
----------
    Chen et al. *Low-dose CT with a residual encoder-decoder convolutional
    neural network.* IEEE TMI 36(12), 2017.
    https://github.com/SSinyu/RED-CNN

Usage
-----
    conda run -n tensor python baselines/redcnn/train.py
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extern" / "RED-CNN"))

from networks import RED_CNN                                   # noqa: E402

from trainer.parallel_fbp import DifferentiableParallelFBP  # noqa: E402
from utils.lodopab_dataset import LoDoPaBDataset                     # noqa: E402
from utils.metrics import compute_psnr                                # noqa: E402

# ── Hyperparameters (verbatim from upstream solver.py / main.py) ──────────────
NUM_EPOCHS    = 100
# BATCH_SIZE bumped from upstream 16 → 64 to use the 16 GB GPU efficiently
# (upstream targeted ~6 GB cards). Effective per-step patches = BATCH_SIZE × PATCH_N
# = 640. lr is unchanged (gradient noise from 4× batch is mild for MSE on patches).
BATCH_SIZE    = 64
PATCH_N       = 10
PATCH_SIZE    = 64
LR            = 1e-5
DECAY_ITERS   = 3000
SAVE_ITERS    = 1000
PRINT_ITERS   = 20
NUM_WORKERS   = 8
ARTIFACT_SEED = None         # train: fresh random artifacts every step

# ── Validation (added on top of upstream — best-checkpoint tracking) ─────────
VAL_SAMPLES        = 200      # fixed val subset (matches baselines/bm3d_eval)
VAL_ARTIFACT_SEED  = 42       # deterministic val artifacts across all baselines
VAL_BATCH_SIZE     = 1
DATA_RANGE         = 1.0

# ── LoDoPaB-specific paths ────────────────────────────────────────────────────
SAVE_DIR = ROOT / "checkpoints" / "baselines" / "redcnn"
LOG_DIR  = ROOT / "output"      / "baselines" / "redcnn" / "log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger() -> logging.Logger:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_{ts}.log"
    logger = logging.getLogger("redcnn_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path);  fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler();        sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"Log: {log_path}")
    return logger


def get_patches(noisy: torch.Tensor, gt: torch.Tensor,
                patch_n: int, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crop ``patch_n`` patches per slice — mirrors loader.py:get_patch.

    Inputs  : (B, 1, H, W) on GPU.  Outputs: (B*patch_n, 1, ps, ps).
    """
    B, _, H, W = noisy.shape
    out_n = torch.empty((B, patch_n, 1, patch_size, patch_size),
                        device=noisy.device, dtype=noisy.dtype)
    out_g = torch.empty_like(out_n)
    for b in range(B):
        for k in range(patch_n):
            top  = np.random.randint(0, H - patch_size)
            left = np.random.randint(0, W - patch_size)
            out_n[b, k, 0] = noisy[b, 0, top:top+patch_size, left:left+patch_size]
            out_g[b, k, 0] = gt   [b, 0, top:top+patch_size, left:left+patch_size]
    return (out_n.view(-1, 1, patch_size, patch_size),
            out_g.view(-1, 1, patch_size, patch_size))


def main():
    logger = setup_logger()
    logger.info(f"Device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = LoDoPaBDataset("train", None,
                              add_artifacts=True, artifact_seed=ARTIFACT_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    logger.info(f"Train slices: {len(train_ds)}  | iters/epoch ≈ {len(train_loader)}")

    full_val_ds = LoDoPaBDataset("validation", None,
                                 add_artifacts=True,
                                 artifact_seed=VAL_ARTIFACT_SEED)
    val_idx     = np.random.default_rng(seed=0).choice(
        len(full_val_ds), size=min(VAL_SAMPLES, len(full_val_ds)),
        replace=False).tolist()
    val_ds      = Subset(full_val_ds, val_idx)
    val_loader  = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    logger.info(f"Val slices: {len(val_ds)} (subset of {len(full_val_ds)}, seed=42)")

    fbp = DifferentiableParallelFBP.from_geometry(train_ds.geometry).to(DEVICE)
    fbp.eval()
    for p in fbp.parameters():
        p.requires_grad_(False)

    # ── Model / optimiser (upstream Solver) ───────────────────────────────────
    model = RED_CNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"RED_CNN: {n_params:,} parameters")

    criterion = nn.MSELoss()
    lr = LR
    optimizer = optim.Adam(model.parameters(), lr=lr)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        psnr_sum = 0.0
        for sino, gt in val_loader:
            sino = sino.to(DEVICE, non_blocking=True)
            gt   = gt  .to(DEVICE, non_blocking=True)
            noisy = fbp(sino).clamp(0.0, 1.0)
            pred  = model(noisy).clamp(0.0, 1.0)
            psnr_sum += compute_psnr(pred, gt, DATA_RANGE)
        model.train()
        return psnr_sum / len(val_loader)

    # ── Train loop (mirrors solver.py:Solver.train + per-epoch validation) ────
    total_iters = 0
    start = time.time()
    train_losses = []
    best_psnr   = -float("inf")
    best_path   = SAVE_DIR / "REDCNN_best.ckpt"

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for it, (sino, gt) in enumerate(train_loader):
            sino = sino.to(DEVICE, non_blocking=True)
            gt   = gt  .to(DEVICE, non_blocking=True)

            with torch.no_grad():
                noisy = fbp(sino)                # (B, 1, 362, 362) in [0,1]
                noisy = noisy.clamp(0.0, 1.0)

            x, y = get_patches(noisy, gt, PATCH_N, PATCH_SIZE)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iters += 1
            train_losses.append(loss.item())

            if total_iters % PRINT_ITERS == 0:
                window = train_losses[-PRINT_ITERS:]
                avg_loss = float(np.mean(window))
                logger.info(
                    f"STEP [{total_iters}] EPOCH [{epoch}/{NUM_EPOCHS}] "
                    f"ITER [{it+1}/{len(train_loader)}] "
                    f"LOSS {loss.item():.6f}  AVG[{PRINT_ITERS}] {avg_loss:.6f}  "
                    f"TIME {time.time()-start:.0f}s"
                )

            if total_iters % DECAY_ITERS == 0:
                lr *= 0.5
                for g in optimizer.param_groups:
                    g["lr"] = lr
                logger.info(f"  lr decayed → {lr:.2e}")

            if total_iters % SAVE_ITERS == 0:
                ckpt = SAVE_DIR / f"REDCNN_{total_iters}iter.ckpt"
                torch.save(model.state_dict(), ckpt)
                np.save(SAVE_DIR / f"loss_{total_iters}_iter.npy",
                        np.array(train_losses, dtype=np.float32))
                logger.info(f"  saved: {ckpt.name}")

        # ── End-of-epoch validation + best-checkpoint tracking ────────────────
        val_psnr = validate()
        epoch_iters = len(train_loader)
        epoch_losses = train_losses[-epoch_iters:]
        train_avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        delta = val_psnr - best_psnr if best_psnr != -float("inf") else 0.0
        marker = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), best_path)
            marker = "  ★ new best"
        logger.info(
            f"Epoch {epoch:03d} | train_loss(avg) {train_avg:.6f}  "
            f"val PSNR {val_psnr:.4f} dB  Δ {delta:+.4f}  "
            f"(best {best_psnr:.4f}){marker}"
        )

    # final save
    final = SAVE_DIR / f"REDCNN_{total_iters}iter.ckpt"
    torch.save(model.state_dict(), final)
    np.save(SAVE_DIR / f"loss_{total_iters}_iter.npy",
            np.array(train_losses, dtype=np.float32))
    logger.info(f"Done.  Final: {final}")
    logger.info(f"Best val PSNR: {best_psnr:.4f} dB  ({best_path})")


if __name__ == "__main__":
    main()
