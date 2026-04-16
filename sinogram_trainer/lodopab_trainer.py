"""
LoDoPaBTrainer
==============
Progressive cascaded sinogram denoising trainer for LoDoPaB-CT.

Loss strategy
-------------
LoDoPaB provides no clean GT sinogram — only (noisy_sino, gt_img).
A sinogram-domain loss against noisy_sino would teach the model to
preserve noise; a pseudo-GT sinogram from forward-projecting gt_img
would have a value scale mismatch (~7.6×) because the dataset was
generated from a 1000×1000 upsampled image, not 362×362.

The only principled supervision is therefore image-domain only:

    loss = Stage0Loss( FBP(pred_sino), gt_img )   ← SSIM + L1 in image space

Gradients flow back through the differentiable FBP into the sinogram
denoiser weights automatically.

Stage 0  — Primary sinogram denoiser (UNet, sigmoid output)
    input  : noisy sinogram  (B, 1, 1000, 513)
    output : denoised sinogram  ∈ [0, 1]
    loss   : Stage0Loss( FBP(pred_sino), gt_img )

Stage 1+ — Residual corrector (ResUNet, tanh output)
    input  : Stage 0 output (frozen)
    output : additive residual; final = clamp(stage0_out + residual, 0, 1)
    loss   : Stage0Loss( FBP(final_sino), gt_img )
"""

import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from base import BaseTrainer
from models.image_domain.unet import UNet
from models.image_domain.resunet import ResUNet
from utils.lodopab_dataset import LoDoPaBDataset
from utils.loss import Stage0Loss
from utils.metrics import compute_psnr, compute_ssim, compute_rmse
from utils.help import EarlyStopping
from sinogram_trainer.parallel_fbp import DifferentiableParallelFBP


class LoDoPaBTrainer(BaseTrainer):
    """Progressive cascaded sinogram denoising trainer for LoDoPaB-CT."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_stages = self.config["num_stages"]

        self.logger.info("=" * 80)
        self.logger.info(f"LoDoPaB Cascaded Sinogram Trainer — {self.config['test_no']}")
        self.logger.info("=" * 80)
        self.logger.info(f"Stages: {self.num_stages}")
        self.logger.info("Loss: Stage0Loss (SSIM + L1) on FBP-reconstructed image only")

        self.setup_data()
        self.setup_models()
        self.setup_losses()
        self.setup_optimizers()

        self.stage_train_losses = [[] for _ in range(self.num_stages)]
        self.stage_val_losses   = [[] for _ in range(self.num_stages)]
        self.stage_best_epochs  = [0]  * self.num_stages

    # ── Data ─────────────────────────────────────────────────────────────────

    def setup_data(self):
        self.logger.info("Setting up LoDoPaB datasets …")

        data_path = self.config.get("data_path", None)

        # Sinograms are stored as  −ln(Ñ/N₀) / μ_max  (Leuschner et al. 2021,
        # Fig. 4 line 13), so values already lie in [0, ~0.13] ⊂ [0, 1].
        # Additional z-score normalisation would map those values to [−1.4, 4.35],
        # causing the sigmoid output of Stage 0 to cover only ~41 % of the true
        # sinogram range after denormalisation.  We therefore use the sinograms
        # as-is — no norm_mean / norm_std passed to the dataset.
        self.logger.info("Sinogram normalisation: none (μ_max-normalised by dataset, range [0, ~0.13])")

        train_ds = LoDoPaBDataset("train",      data_path)
        val_ds   = LoDoPaBDataset("validation", data_path)
        test_ds  = LoDoPaBDataset("test",       data_path)

        if self.test_local:
            train_ds = Subset(train_ds, range(32))
            val_ds   = Subset(val_ds,   range(8))
            test_ds  = Subset(test_ds,  range(8))

        self.full_train_dataset = train_ds

        loader_kw = dict(
            batch_size=self.config["batch_size"],
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
        )
        shared_train = DataLoader(train_ds, shuffle=True,  **loader_kw)
        shared_val   = DataLoader(val_ds,   shuffle=False, **loader_kw)

        # Every stage uses the same loaders
        self.stage_train_loaders = [shared_train] * self.num_stages
        self.stage_val_loaders   = [shared_val]   * self.num_stages

        # Single-sample loaders for cascade evaluation and test
        self.val_loader  = DataLoader(val_ds,  batch_size=1, shuffle=False,
                                      num_workers=loader_kw["num_workers"], pin_memory=True)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                      num_workers=loader_kw["num_workers"], pin_memory=True)

        self.logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # Store geometry for FBP setup
        raw = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        self._geometry = raw.geometry

    # ── Models ────────────────────────────────────────────────────────────────

    def setup_models(self):
        self.logger.info("Setting up stage models …")
        self.models = []

        for stage in range(self.num_stages):
            if stage == 0:
                features   = self.config.get("stage0_features", [32, 64, 128])
                activation = "sigmoid"
                model = UNet(
                    in_channels=1, out_channels=1,
                    features=features, output_activation=activation,
                ).to(self.device)
            else:
                features   = self.config.get("stage1_features", [16, 32, 64, 128, 256])
                activation = "tanh"
                model = ResUNet(
                    in_channels=1, out_channels=1,
                    features=features, output_activation=activation,
                ).to(self.device)

            self.models.append(model)
            n = sum(p.numel() for p in model.parameters())
            self.logger.info(f"  Stage {stage}: {n:,} parameters  (activation={activation})")

        total = sum(p.numel() for m in self.models for p in m.parameters())
        self.logger.info(f"  Total: {total:,} parameters")

        # Differentiable parallel-beam FBP
        self.fbp = DifferentiableParallelFBP.from_geometry(self._geometry).to(self.device)
        self.logger.info(
            f"  FBP: parallel beam | {self._geometry['num_angles']} angles | "
            f"{self._geometry['num_detectors']} detectors | "
            f"recon {self._geometry['image_size']}×{self._geometry['image_size']}"
        )

    # ── Losses ────────────────────────────────────────────────────────────────

    def setup_losses(self):
        self.logger.info("Setting up loss functions …")
        # Single image-domain loss used for every stage.
        # Sinogram-domain loss is dropped: we have no clean GT sinogram —
        # only gt_img.  The differentiable FBP backpropagates image gradients
        # into the sinogram denoiser weights automatically.
        self.img_loss_fn = Stage0Loss().to(self.device)
        self.logger.info("  Stage0Loss (SSIM + L1) on FBP(pred_sino) vs gt_img — all stages\n")

    # ── Optimizers ────────────────────────────────────────────────────────────

    def setup_optimizers(self):
        self.optimizers = []
        self.schedulers = []
        for stage in range(self.num_stages):
            opt = optim.AdamW(self.models[stage].parameters(), lr=self.config["lr"])
            sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
            self.optimizers.append(opt)
            self.schedulers.append(sched)

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _denorm(self, sino: torch.Tensor) -> torch.Tensor:
        """Pass-through: sinograms are already in μ_max-normalised physical scale."""
        return sino

    # ── Cascade helpers ───────────────────────────────────────────────────────

    def _forward_previous_stages(self, noisy_sino: torch.Tensor, up_to: int) -> torch.Tensor:
        """Run noisy_sino through stages 0 … up_to−1 (all frozen)."""
        if up_to == 0:
            return noisy_sino
        with torch.no_grad():
            x = noisy_sino
            for s in range(up_to):
                self.models[s].eval()
                x = torch.clamp(self.models[s](x), 0.0, 1.0)
        return x

    # ── Single-epoch train / validate ─────────────────────────────────────────

    def train_single_epoch(self, stage: int, epoch: int):
        model     = self.models[stage]
        optimizer = self.optimizers[stage]

        model.train()
        running            = {"total": 0.0}
        running_components: dict = {}
        loader = self.stage_train_loaders[stage]
        loop   = tqdm(loader, desc=f"Stage {stage} Epoch {epoch:03d}")

        for noisy_sino, gt_img in loop:
            noisy_sino = noisy_sino.to(self.device)   # (B,1,1000,513)
            gt_img     = gt_img.to(self.device)       # (B,1,362,362)

            optimizer.zero_grad()

            if stage == 0:
                # ── Stage 0: predict denoised sinogram ───────────────────────
                pred_sino = model(noisy_sino)           # (B,1,A,D)  sigmoid → [0,1]
            else:
                # ── Stage 1+: predict additive residual ──────────────────────
                prev_sino = self._forward_previous_stages(noisy_sino, stage)
                residual  = model(prev_sino)            # (B,1,A,D)  tanh → [-1,1]
                pred_sino = torch.clamp(prev_sino + residual, 0.0, 1.0)

            # Sole supervision signal: image-domain Stage0Loss (SSIM + L1)
            # on the FBP reconstruction vs clean GT image.
            # Gradients flow back through the differentiable FBP automatically.
            recon_img = self.fbp(self._denorm(pred_sino))   # (B,1,H,W)
            loss      = self.img_loss_fn(recon_img, gt_img)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running["total"] += loss.item()
            for k, v in self.img_loss_fn.components.items():
                running_components[k] = running_components.get(k, 0.0) + v
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        n = len(loader)
        return (
            {k: v / n for k, v in running.items()},
            {k: v / n for k, v in running_components.items()},
        )

    @torch.no_grad()
    def validate_single_epoch(self, stage: int, epoch: int):
        model = self.models[stage]
        model.eval()

        running            = {"total": 0.0, "psnr": 0.0, "ssim": 0.0}
        running_components: dict = {}

        for noisy_sino, gt_img in tqdm(self.stage_val_loaders[stage],
                                        desc=f"Val Stage {stage} Epoch {epoch:03d}"):
            noisy_sino = noisy_sino.to(self.device)
            gt_img     = gt_img.to(self.device)

            if stage == 0:
                pred_sino = torch.clamp(model(noisy_sino), 0.0, 1.0)
            else:
                prev_sino = self._forward_previous_stages(noisy_sino, stage)
                residual  = model(prev_sino)
                pred_sino = torch.clamp(prev_sino + residual, 0.0, 1.0)

            recon_img = self.fbp(self._denorm(pred_sino))
            loss      = self.img_loss_fn(recon_img, gt_img)

            running["total"] += loss.item()
            for k, v in self.img_loss_fn.components.items():
                running_components[k] = running_components.get(k, 0.0) + v

            dr = (gt_img.max() - gt_img.min()).item() + 1e-8
            running["psnr"] += float(compute_psnr(recon_img, gt_img, data_range=dr))
            running["ssim"] += float(compute_ssim(recon_img, gt_img, data_range=dr))

        n = len(self.stage_val_loaders[stage])
        metrics = {k: v / n for k, v in running.items()}
        metrics["components"] = {k: v / n for k, v in running_components.items()}
        return metrics

    # ── Stage training ────────────────────────────────────────────────────────

    def train_stage(self, stage: int):
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Training Stage {stage}")
        self.logger.info("=" * 80)

        stopper       = EarlyStopping(patience=self.config["patience"], min_delta=0.0)
        best_val_loss = float("inf")

        if stage == 0 and "max_stage0_epochs" in self.config:
            max_epochs = min(self.config["epochs"], self.config["max_stage0_epochs"])
            self.logger.info(f"Stage 0 epoch cap: {max_epochs}")
        else:
            max_epochs = self.config["epochs"]

        for epoch in range(max_epochs):
            tr, tr_comp = self.train_single_epoch(stage, epoch)
            va          = self.validate_single_epoch(stage, epoch)

            self.schedulers[stage].step(va["total"])

            tr_comp_str = "  ".join(f"{k}={v:.4f}" for k, v in tr_comp.items())
            va_comp_str = "  ".join(f"{k}={v:.4f}" for k, v in va["components"].items())

            self.logger.info(
                f"Stage {stage} Epoch {epoch:03d} | "
                f"Train {tr['total']:.4f} | "
                f"Val {va['total']:.4f} | "
                f"PSNR {va['psnr']:.2f} SSIM {va['ssim']:.4f}"
            )
            self.logger.info(f"  Train img components: {tr_comp_str}")
            self.logger.info(f"  Val   img components: {va_comp_str}")

            self.stage_train_losses[stage].append(tr["total"])
            self.stage_val_losses[stage].append(va["total"])

            stopper.check_early_stop(va["total"])

            if va["total"] < best_val_loss:
                best_val_loss = va["total"]
                self._save_stage_model(stage)
                self.stage_best_epochs[stage] = epoch
                self.logger.info(f"  ✓ Best model saved for stage {stage}")

            if stopper.stop_training:
                self.logger.info(f"  Early stopping at epoch {epoch}")
                break

        self._load_stage_model(stage)
        self.logger.info(f"Loaded best stage {stage} model (epoch {self.stage_best_epochs[stage]})")
        self.logger.info("=" * 80 + "\n")

    # ── Full training ─────────────────────────────────────────────────────────

    def train(self):
        self.logger.info("\n" + "#" * 80)
        self.logger.info("STARTING LODOPAB PROGRESSIVE CASCADED TRAINING")
        self.logger.info("#" * 80 + "\n")

        for stage in range(self.num_stages):
            self.train_stage(stage)

            self.logger.info(f"\nFull-cascade validation after stage {stage} …")
            metrics = self._validate_full_cascade(up_to=stage + 1)
            self.logger.info(
                f"  Cascade stages 0–{stage}: "
                f"PSNR {metrics['psnr']:.2f} | SSIM {metrics['ssim']:.4f}"
            )

        self.logger.info("\n" + "#" * 80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("#" * 80 + "\n")
        self._plot_stage_loss_curves()

    # ── Full-cascade validation ───────────────────────────────────────────────

    @torch.no_grad()
    def _validate_full_cascade(self, up_to: int = None):
        if up_to is None:
            up_to = self.num_stages

        for m in self.models[:up_to]:
            m.eval()

        psnr_sum = ssim_sum = 0.0

        for noisy_sino, gt_img in self.val_loader:
            noisy_sino = noisy_sino.to(self.device)
            gt_img     = gt_img.to(self.device)

            # Stage 0
            x = torch.clamp(self.models[0](noisy_sino), 0.0, 1.0)
            # Stage 1+
            for s in range(1, up_to):
                residual = self.models[s](x)
                x = torch.clamp(x + residual, 0.0, 1.0)

            recon = self.fbp(self._denorm(x))
            dr    = (gt_img.max() - gt_img.min()).item() + 1e-8
            psnr_sum += float(compute_psnr(recon, gt_img, data_range=dr))
            ssim_sum += float(compute_ssim(recon, gt_img, data_range=dr))

        n = len(self.val_loader)
        return {"psnr": psnr_sum / n, "ssim": ssim_sum / n}

    # ── Testing ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TESTING FULL CASCADE")
        self.logger.info("=" * 80)

        for stage in range(self.num_stages):
            self._load_stage_model(stage)
            self.models[stage].eval()

        baseline_psnr = baseline_ssim = baseline_rmse = 0.0
        stage_psnr = [0.0] * self.num_stages
        stage_ssim = [0.0] * self.num_stages
        stage_rmse = [0.0] * self.num_stages

        n_test = len(self.test_loader)
        rng = np.random.default_rng(seed=42)
        save_idx = set(rng.choice(n_test, size=min(10, n_test), replace=False).tolist())
        self.logger.info(f"Saving visualisations for indices: {sorted(save_idx)}")

        for i, (noisy_sino, gt_img) in enumerate(tqdm(self.test_loader, desc="Testing")):
            noisy_sino = noisy_sino.to(self.device)
            gt_img     = gt_img.to(self.device)

            # Baseline: FBP of the noisy sinogram
            baseline_recon = self.fbp(self._denorm(noisy_sino))
            dr = (gt_img.max() - gt_img.min()).item() + 1e-8
            baseline_psnr += float(compute_psnr(baseline_recon, gt_img, data_range=dr))
            baseline_ssim += float(compute_ssim(baseline_recon, gt_img, data_range=dr))
            baseline_rmse += float(compute_rmse(baseline_recon, gt_img))

            # Cascade
            stage_recons = []
            x = torch.clamp(self.models[0](noisy_sino), 0.0, 1.0)
            recon_0 = self.fbp(self._denorm(x))
            stage_recons.append(recon_0)
            stage_psnr[0] += float(compute_psnr(recon_0, gt_img, data_range=dr))
            stage_ssim[0] += float(compute_ssim(recon_0, gt_img, data_range=dr))
            stage_rmse[0] += float(compute_rmse(recon_0, gt_img))

            for s in range(1, self.num_stages):
                residual = self.models[s](x)
                x = torch.clamp(x + residual, 0.0, 1.0)
                recon_s = self.fbp(self._denorm(x))
                stage_recons.append(recon_s)
                stage_psnr[s] += float(compute_psnr(recon_s, gt_img, data_range=dr))
                stage_ssim[s] += float(compute_ssim(recon_s, gt_img, data_range=dr))
                stage_rmse[s] += float(compute_rmse(recon_s, gt_img))

            if i in save_idx:
                self._save_test_image(i, baseline_recon, stage_recons, gt_img)

        baseline_psnr /= n_test
        baseline_ssim /= n_test
        baseline_rmse /= n_test
        stage_psnr = [v / n_test for v in stage_psnr]
        stage_ssim = [v / n_test for v in stage_ssim]
        stage_rmse = [v / n_test for v in stage_rmse]

        self.logger.info(
            f"\n[Baseline FBP(noisy)] PSNR: {baseline_psnr:.2f} | "
            f"SSIM: {baseline_ssim:.4f} | RMSE: {baseline_rmse:.4f}"
        )
        for s in range(self.num_stages):
            gain_psnr = stage_psnr[s] - baseline_psnr
            gain_ssim = stage_ssim[s] - baseline_ssim
            drop_rmse = baseline_rmse - stage_rmse[s]
            self.logger.info(
                f"[Stage {s}] PSNR: {stage_psnr[s]:.2f} (+{gain_psnr:.2f}) | "
                f"SSIM: {stage_ssim[s]:.4f} (+{gain_ssim:.4f}) | "
                f"RMSE: {stage_rmse[s]:.4f} ({drop_rmse:.4f}↓)"
            )
        self.logger.info("=" * 80 + "\n")

        self.test_results = {
            "baseline": {"psnr": baseline_psnr, "ssim": baseline_ssim, "rmse": baseline_rmse},
            "stages": {"psnr": stage_psnr, "ssim": stage_ssim, "rmse": stage_rmse},
        }
        self._plot_stage_improvement()

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_stage_model(self, stage: int):
        save_dir = os.path.join(
            self.config["checkpoints_dir"], self.config["model"], self.config["test_no"]
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.models[stage].state_dict(),
            os.path.join(save_dir, f"stage_{stage}_best.pth"),
        )

    def _load_stage_model(self, stage: int):
        path = os.path.join(
            self.config["checkpoints_dir"],
            self.config["model"],
            self.config["test_no"],
            f"stage_{stage}_best.pth",
        )
        self.models[stage].load_state_dict(torch.load(path, weights_only=True))

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _save_test_image(self, idx, baseline_recon, stage_recons, gt_img):
        n_plots = 2 + len(stage_recons)   # FBP baseline + stages + GT
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        # Display window from Leuschner et al. (2021) Fig. 5: [0, 0.45]
        vmin, vmax = 0.0, 0.45

        dr = (gt_img.max() - gt_img.min()).item() + 1e-8

        def _to_display(img_t):
            """Convert tensor to display array with orientation correction.

            GT is stored transposed (ODL x-y convention, per paper Fig. 4).
            FBP outputs from diffct use y-downward convention vs ODL y-upward,
            so they need an additional vertical flip on top of the transpose.
            Both cases end up using flipud(arr.T) since the GT tensor is already
            in ODL space (as it came from the same HDF5 file).
            """
            arr = img_t.cpu().squeeze().numpy()
            return np.flipud(arr.T)

        def _show(ax, img_t, title):
            ax.imshow(_to_display(img_t), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # FBP baseline
        fbp_psnr = float(compute_psnr(baseline_recon, gt_img, data_range=dr))
        fbp_ssim = float(compute_ssim(baseline_recon, gt_img, data_range=dr))
        _show(axes[0], baseline_recon,
              f"FBP (noisy)\nPSNR: {fbp_psnr:.2f}\nSSIM: {fbp_ssim:.4f}")

        # Each cascade stage
        for s, recon in enumerate(stage_recons):
            s_psnr = float(compute_psnr(recon, gt_img, data_range=dr))
            s_ssim = float(compute_ssim(recon, gt_img, data_range=dr))
            _show(axes[s + 1], recon,
                  f"Stage {s}\nPSNR: {s_psnr:.2f}\nSSIM: {s_ssim:.4f}")

        # Ground truth — reference, no metrics
        _show(axes[-1], gt_img, "Ground Truth")

        plt.tight_layout()
        save_dir = os.path.join(
            self.config["output_dir"], self.config["model"],
            "fig", self.config["test_no"], "results",
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"result_{idx:04d}.png"), dpi=150)
        plt.close(fig)

    def _plot_stage_loss_curves(self):
        fig, axes = plt.subplots(1, self.num_stages, figsize=(6 * self.num_stages, 5))
        if self.num_stages == 1:
            axes = [axes]

        for s in range(self.num_stages):
            epochs = range(1, len(self.stage_train_losses[s]) + 1)
            axes[s].plot(epochs, self.stage_train_losses[s], label="Train", marker="o")
            axes[s].plot(epochs, self.stage_val_losses[s],   label="Val",   marker="s")
            axes[s].axvline(self.stage_best_epochs[s] + 1, linestyle="--", color="red")
            axes[s].set_title(f"Stage {s} Loss")
            axes[s].set_xlabel("Epoch")
            axes[s].set_ylabel("Loss")
            axes[s].legend()
            axes[s].grid(True)

        plt.tight_layout()
        save_dir = os.path.join(
            self.config["output_dir"], self.config["model"], "fig", self.config["test_no"]
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "stage_loss_curves.png"), dpi=150)
        plt.close()
        self.logger.info(f"Loss curves saved → {save_dir}")

    def _plot_stage_improvement(self):
        b = self.test_results["baseline"]
        s = self.test_results["stages"]

        labels = ["Noisy FBP"] + [f"S{i}" for i in range(self.num_stages)]
        psnr   = [b["psnr"]] + s["psnr"]
        ssim   = [b["ssim"]] + s["ssim"]
        rmse   = [b["rmse"]] + s["rmse"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, vals, ylabel in zip(axes, [psnr, ssim, rmse], ["PSNR", "SSIM", "RMSE"]):
            ax.plot(labels, vals, marker="o")
            ax.set_title(f"{ylabel} Improvement")
            ax.set_xlabel("Stage")
            ax.set_ylabel(ylabel)
            ax.grid(True)

        plt.tight_layout()
        save_dir = os.path.join(
            self.config["output_dir"], self.config["model"], "fig", self.config["test_no"]
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "stage_improvement.png"), dpi=150)
        plt.close()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.train()
        self.test()

    def validate(self):
        return self._validate_full_cascade()
