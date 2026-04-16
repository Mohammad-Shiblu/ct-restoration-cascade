"""
SinogramTrainer
===============
End-to-end sinogram denoising with diffCT differentiable FBP.

Architecture
------------
  noisy_sino (B,1,A,D) ──► DnCNN ──► residual
                                │
                  pred_sino = noisy_sino − residual
                                │
              denormalise → physical attenuation units
                                │
                     diffCT FBP (differentiable)
                                │
                         pred_img (B,1,H,W)

Loss
----
  L = w_sino · SinoLoss(pred_sino, fd_sino)
    + w_img  · L1(FBP(pred_phys), FBP(fd_phys).detach())

  SinoLoss = 0.5 · (1−SSIM) + 0.5 · L1   (Stage0Loss from utils/loss.py)

Gradients flow from the image-domain L1 through the differentiable FBP
back into the DnCNN weights, enforcing structural quality of the
reconstruction and not just pixel-wise sinogram accuracy.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.sinogram_domain.dncnn import ResidualCorrectionNet
from utils.sinogram_dataset import SinogramDataset
from sinogram_trainer.fbp import DifferentiableFBP, radon_fbp
from utils.loss import Stage0Loss
from utils.metrics import compute_psnr, compute_ssim


# ---------------------------------------------------------------------------
# Minimal early-stopping (self-contained so the package stays independent)
# ---------------------------------------------------------------------------

class _EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.stop_training = False

    def check(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SinogramTrainer:
    """End-to-end sinogram denoising pipeline trainer."""

    def __init__(self, config: dict, logger, test_local: bool = False):
        self.config = config
        self.device = config["device"]
        self.logger = logger
        self.test_local = test_local
        self.train_losses: list = []
        self.val_losses: list = []

        self.setup_data()
        self.setup_models()
        self.setup_losses()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    # ── Data ─────────────────────────────────────────────────────────────────

    def setup_data(self):
        cfg = self.config
        manifest = cfg["manifest"]
        root = "."

        def _make_ds(mode, norm_mean=None, norm_std=None):
            return SinogramDataset(
                manifest_path=manifest,
                root_dir=root,
                mode=mode,
                val_id=cfg["val_id"],
                test_id=cfg["test_id"],
                angle_subsample=cfg.get("angle_subsample", 1),
                norm_mean=norm_mean,
                norm_std=norm_std,
            )

        train_ds_raw = _make_ds("train")
        norm_mean, norm_std = train_ds_raw.compute_norm_stats()
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.logger.info(
            f"Sinogram norm stats — mean: {norm_mean:.4f}, std: {norm_std:.4f}"
        )

        train_ds = _make_ds("train", norm_mean, norm_std)
        val_ds   = _make_ds("val",   norm_mean, norm_std)
        test_ds  = _make_ds("test",  norm_mean, norm_std)

        self._geo = train_ds.geometry

        if self.test_local:
            train_ds = Subset(train_ds, range(min(4, len(train_ds))))
            val_ds   = Subset(val_ds,   range(min(2, len(val_ds))))
            test_ds  = Subset(test_ds,  range(min(2, len(test_ds))))

        loader_kw = dict(
            batch_size=cfg["batch_size"],
            num_workers=cfg.get("num_workers", 0),
            pin_memory=(self.device == "cuda"),
        )
        self.train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
        self.val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
        self.test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

        self.logger.info(
            f"Dataset — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
        )

    # ── Models ────────────────────────────────────────────────────────────────

    def setup_models(self):
        cfg = self.config

        self.model = ResidualCorrectionNet(
            in_channels=1,
            num_layers=cfg.get("num_layers", 10),
            num_features=cfg.get("num_features", 64),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Sinogram DnCNN — {n_params:,} parameters")

        geo      = self._geo
        rotview  = int(geo.get("rotview", 2304))
        subsample = cfg.get("angle_subsample", 1)
        num_angles = math.ceil(rotview / subsample)

        self.fbp = DifferentiableFBP.from_geometry(
            geo=geo,
            num_angles=num_angles,
            H=cfg["recon_height"],
            W=cfg["recon_width"],
            voxel_spacing=cfg.get("voxel_spacing", 0.7),
            angle_subsample=subsample,
        ).to(self.device)

        self.logger.info(
            f"FBP — sdd={self.fbp.sdd:.1f}, sid={self.fbp.sid:.1f}, "
            f"du={self.fbp.du:.4f}, num_angles={num_angles}, "
            f"det_offset={self.fbp.detector_offset:.4f}, "
            f"recon=({cfg['recon_height']}×{cfg['recon_width']}), "
            f"voxel={self.fbp.voxel_spacing:.4f} mm"
        )

    # ── Losses ────────────────────────────────────────────────────────────────

    def setup_losses(self):
        cfg = self.config
        self.sino_loss_fn = Stage0Loss(w_ssim=0.5, w_l1=0.5).to(self.device)
        self.img_loss_fn  = nn.L1Loss()
        self.w_sino = cfg.get("sino_loss_weight", 0.6)
        self.w_img  = cfg.get("image_loss_weight", 0.4)

    def _total_loss(self, pred_sino, fd_sino, pred_img, target_img):
        """Combined sinogram + image domain loss."""
        l_sino = self.sino_loss_fn(pred_sino, fd_sino)
        l_img  = self.img_loss_fn(pred_img, target_img)
        return self.w_sino * l_sino + self.w_img * l_img

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _denorm(self, sino_norm: torch.Tensor) -> torch.Tensor:
        """Normalised tensor → physical log-attenuation."""
        return sino_norm * (self.norm_std + 1e-8) + self.norm_mean

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        cfg     = self.config
        stopper = _EarlyStopping(patience=cfg["patience"])

        self.logger.info(
            f"{cfg['test_no']}: optimizer=AdamW lr={cfg['lr']}, "
            f"w_sino={self.w_sino}, w_img={self.w_img}"
        )

        for epoch in range(cfg["epochs"]):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
            for noisy, fd in pbar:
                noisy = noisy.to(self.device)
                fd    = fd.to(self.device)

                residual  = self.model(noisy)
                pred_sino = noisy - residual

                pred_img = self.fbp(self._denorm(pred_sino))
                with torch.no_grad():
                    target_img = self.fbp(self._denorm(fd))

                self.optimizer.zero_grad()
                loss = self._total_loss(pred_sino, fd, pred_img, target_img)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= len(self.train_loader)
            val_metrics = self.validate()
            val_loss = val_metrics[0]

            self.scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.logger.info(
                f"Epoch {epoch+1:03d} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"sino_psnr={val_metrics[1]:.2f} sino_ssim={val_metrics[2]:.4f} | "
                f"img_psnr={val_metrics[3]:.2f}  img_ssim={val_metrics[4]:.4f}"
            )

            stopper.check(val_loss)
            if stopper.counter == 0:
                self.save_model()
                self.logger.info(f"  ↳ best model saved (val_loss={val_loss:.4f})")
            if stopper.stop_training:
                self.logger.info(f"Early stop at epoch {epoch+1}")
                break

        self.plot_loss_curves()

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self):
        self.model.eval()
        tot_loss = s_psnr = s_ssim = i_psnr = i_ssim = 0.0

        with torch.no_grad():
            for noisy, fd in self.val_loader:
                noisy = noisy.to(self.device)
                fd    = fd.to(self.device)

                pred_sino  = noisy - self.model(noisy)
                pred_img   = self.fbp(self._denorm(pred_sino))
                target_img = self.fbp(self._denorm(fd))

                tot_loss += self._total_loss(pred_sino, fd, pred_img, target_img).item()

                dr_sino = (fd.max() - fd.min()).item() + 1e-8
                s_psnr += compute_psnr(pred_sino, fd, data_range=dr_sino)
                s_ssim += compute_ssim(pred_sino, fd, data_range=dr_sino)

                dr_img = (target_img.max() - target_img.min()).item() + 1e-8
                i_psnr += compute_psnr(pred_img, target_img, data_range=dr_img)
                i_ssim += compute_ssim(pred_img, target_img, data_range=dr_img)

        n = len(self.val_loader)
        return tot_loss / n, s_psnr / n, s_ssim / n, i_psnr / n, i_ssim / n

    # ── Testing ──────────────────────────────────────────────────────────────

    def test(self):
        self.load_model()
        self.logger.info("Best checkpoint loaded for testing.")
        self.model.eval()

        geo       = self._geo
        subsample = self.config.get("angle_subsample", 1)
        recon_h   = self.config["recon_height"]
        vox       = self.config.get("voxel_size", 0.7)

        ori_sp = ori_ss = pred_sp = pred_ss = 0.0
        ori_ip = ori_ii = pred_ip = pred_ii = 0.0
        n_batches = n_samples = 0

        with torch.no_grad():
            for i, (noisy, fd) in enumerate(self.test_loader):
                noisy = noisy.to(self.device)
                fd    = fd.to(self.device)

                pred_sino = noisy - self.model(noisy)

                dr_sino = (fd.max() - fd.min()).item() + 1e-8
                ori_sp  += compute_psnr(noisy,     fd, data_range=dr_sino)
                ori_ss  += compute_ssim(noisy,     fd, data_range=dr_sino)
                pred_sp += compute_psnr(pred_sino, fd, data_range=dr_sino)
                pred_ss += compute_ssim(pred_sino, fd, data_range=dr_sino)
                n_batches += 1

                for b in range(noisy.shape[0]):
                    n_phys = self._denorm(noisy[b]).squeeze(0).cpu().numpy()
                    p_phys = self._denorm(pred_sino[b]).squeeze(0).cpu().numpy()
                    f_phys = self._denorm(fd[b]).squeeze(0).cpu().numpy()

                    n_img = radon_fbp(n_phys, geo, subsample, recon_h, vox, device=self.device)
                    p_img = radon_fbp(p_phys, geo, subsample, recon_h, vox, device=self.device)
                    t_img = radon_fbp(f_phys, geo, subsample, recon_h, vox, device=self.device)

                    dr_img  = float(np.percentile(t_img, 99)) + 1e-8
                    ori_ip  += compute_psnr(n_img, t_img, data_range=dr_img)
                    ori_ii  += compute_ssim(n_img, t_img, data_range=dr_img)
                    pred_ip += compute_psnr(p_img, t_img, data_range=dr_img)
                    pred_ii += compute_ssim(p_img, t_img, data_range=dr_img)

                    sample_idx = i * self.config["batch_size"] + b
                    self._save_test_image(sample_idx, noisy[b], pred_sino[b], fd[b],
                                          n_img, p_img, t_img)
                    n_samples += 1

        self.logger.info(
            f"Sinogram — Noisy  PSNR={ori_sp/n_batches:.2f} SSIM={ori_ss/n_batches:.4f} | "
            f"Pred  PSNR={pred_sp/n_batches:.2f} SSIM={pred_ss/n_batches:.4f}"
        )
        self.logger.info(
            f"Image   — Noisy  PSNR={ori_ip/n_samples:.2f} SSIM={ori_ii/n_samples:.4f} | "
            f"Pred  PSNR={pred_ip/n_samples:.2f} SSIM={pred_ii/n_samples:.4f}"
        )

    # ── I/O helpers ──────────────────────────────────────────────────────────

    def save_model(self):
        ckpt_dir = os.path.join(self.config["checkpoints_dir"], self.config["model"])
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"ckpt_{self.config['test_no']}.pth")
        torch.save(self.model.state_dict(), path)

    def load_model(self):
        path = os.path.join(
            self.config["checkpoints_dir"],
            self.config["model"],
            f"ckpt_{self.config['test_no']}.pth",
        )
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def _save_test_image(self, idx, noisy_n, pred_n, fd_n, n_img, p_img, t_img):
        """
        noisy_n, pred_n, fd_n : Tensor (1, A, D)  normalised sinograms
        n_img, p_img, t_img   : np.ndarray (H, W) from radon_fbp
        """
        out_dir = os.path.join(
            self.config["output_dir"], self.config["model"], "fig",
            self.config["test_no"], "results",
        )
        os.makedirs(out_dir, exist_ok=True)

        noisy_s = noisy_n.cpu().squeeze().numpy()
        pred_s  = pred_n.cpu().squeeze().numpy()
        fd_s    = fd_n.cpu().squeeze().numpy()

        HU_MIN, HU_MAX = -300, 300
        dr = float(t_img.max() - t_img.min()) + 1e-8
        psnr_noisy = compute_psnr(n_img, t_img, data_range=dr)
        psnr_pred  = compute_psnr(p_img, t_img, data_range=dr)
        ssim_noisy = compute_ssim(n_img, t_img, data_range=dr)
        ssim_pred  = compute_ssim(p_img, t_img, data_range=dr)
        diff = p_img - t_img

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle(f"Test sample {idx}  |  window [{HU_MIN}, {HU_MAX}] HU", fontsize=12)

        vmin_s, vmax_s = fd_s.min(), fd_s.max()
        for ax, data, title in zip(
            axes[0, :3],
            [noisy_s, pred_s, fd_s],
            ["Noisy sinogram", "Denoised sinogram", "FD sinogram (target)"],
        ):
            ax.imshow(data, cmap="gray", vmin=vmin_s, vmax=vmax_s, aspect="auto")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        axes[0, 3].axis("off")

        for ax, img, title, metrics in [
            (axes[1, 0], n_img, "FBP(noisy)",
             f"PSNR {psnr_noisy:.2f} dB  SSIM {ssim_noisy:.4f}"),
            (axes[1, 1], p_img, "FBP(denoised)",
             f"PSNR {psnr_pred:.2f} dB  SSIM {ssim_pred:.4f}"),
            (axes[1, 2], t_img, "FBP(FD target)", "reference"),
        ]:
            ax.imshow(img, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(metrics, fontsize=9)
            ax.axis("off")

        abs_max = float(np.abs(diff).max()) + 1e-9
        im_d = axes[1, 3].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        axes[1, 3].set_title("FBP(denoised) − FBP(FD target)", fontsize=11)
        axes[1, 3].set_xlabel(f"max |err| = {abs_max:.1f} HU", fontsize=9)
        axes[1, 3].axis("off")
        plt.colorbar(im_d, ax=axes[1, 3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"result_{idx:04d}.png"), dpi=100, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_loss_curves(self):
        out_dir = os.path.join(
            self.config["output_dir"], self.config["model"], "fig", self.config["test_no"],
        )
        os.makedirs(out_dir, exist_ok=True)

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Train")
        plt.plot(epochs, self.val_losses,   label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Sinogram Pipeline — Loss Curves")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"loss_curves_{self.config['test_no']}.png"))
        plt.close()

    # ── Entry point ──────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("=" * 55)
        self.logger.info("  Sinogram Pipeline — Training")
        self.logger.info("=" * 55)
        self.train()

        self.logger.info("=" * 55)
        self.logger.info("  Sinogram Pipeline — Testing")
        self.logger.info("=" * 55)
        self.test()
