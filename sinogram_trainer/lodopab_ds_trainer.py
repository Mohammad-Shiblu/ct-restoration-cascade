"""
LoDoPaBDeepSupervisionTrainer
==============================
Deep supervision image-domain trainer for LoDoPaB-CT.

Pipeline
--------
    noisy_sino ──→ FBP (no_grad, fixed pre-processor) ──→ noisy_img
                                                               │
                                                          Stage 0 (UNet, sigmoid)
                                                               │ stage0_out  ∈ [0, 1]
                                                          Stage 1 (ResUNet, tanh)
                                                               │ clamp(stage0_out + residual, 0, 1)
                                                              ...

All stages are trained jointly in a single loop (deep supervision).
Each stage output is supervised independently with Stage0Loss (SSIM + L1)
directly against gt_img — no FBP inside the loss because the network
already operates in image domain.

Key differences vs LoDoPaBTrainer (lodopab_trainer.py)
------------------------------------------------------
- FBP is applied once per batch with no_grad as a pre-processor;
  gradients do NOT need to flow back through FBP, which removes the
  most expensive op from the backward pass.
- Stages are trained jointly (deep supervision) instead of sequentially,
  so the model sees all supervision signals in every epoch.

Key differences vs CascadedUnetTrainer (munet_trainer.py)
----------------------------------------------------------
- Input comes from LoDoPaBDataset (noisy_sino, gt_img); FBP converts
  sinograms to the image domain before the network.
- Loss function is Stage0Loss (SSIM + L1) for every stage, following
  the loss design of lodopab_trainer.py.
- Stage 1+ use ResUNet (tanh, additive residual) rather than plain UNet.
"""

import os

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
from utils.loss import Stage0Loss, ResidualRefinementLoss
from utils.metrics import compute_psnr, compute_ssim, compute_rmse
from utils.help import EarlyStopping
from sinogram_trainer.parallel_fbp import DifferentiableParallelFBP


class LoDoPaBDeepSupervisionTrainer(BaseTrainer):
    """Deep supervision image-domain trainer for LoDoPaB-CT."""

    def __init__(self, max_samples: int = None, **kwargs):
        # max_samples can be passed as constructor arg or via config["max_samples"]
        self._max_samples_arg = max_samples
        super().__init__(**kwargs)
        self.max_samples = max_samples if max_samples is not None else self.config.get("max_samples", None)
        self.num_stages = self.config["num_stages"]

        self.logger.info("=" * 80)
        self.logger.info(
            f"LoDoPaB Deep Supervision Trainer — {self.config['test_no']}"
        )
        self.logger.info("=" * 80)
        self.logger.info(f"Stages : {self.num_stages}")
        self.logger.info(
            "Pipeline: noisy_sino → FBP(no_grad) → noisy_img → stages → [out_0, …, out_N]"
        )
        self.logger.info(
            "Loss    : Stage 0 → Stage0Loss (SSIM+L1) | Stage 1+ → ResidualRefinementLoss (L1_resid+SSIM+Grad)"
        )

        self.setup_data()
        self.setup_models()
        self.setup_loss_functions()
        self.setup_optimizers()

    # ── Data ──────────────────────────────────────────────────────────────────

    def setup_data(self):
        self.logger.info("Setting up LoDoPaB datasets …")

        data_path     = self.config.get("data_path", None)
        add_artifacts = self.config.get("add_artifacts", False)
        val_seed      = self.config.get("val_artifact_seed",  42)
        test_seed     = self.config.get("test_artifact_seed",  0)

        train_ds = LoDoPaBDataset("train",      data_path, add_artifacts=add_artifacts)
        val_ds   = LoDoPaBDataset("validation", data_path, add_artifacts=add_artifacts,
                                  artifact_seed=val_seed)
        test_ds  = LoDoPaBDataset("test",       data_path, add_artifacts=add_artifacts,
                                  artifact_seed=test_seed)

        if add_artifacts:
            self.logger.info(
                "Artifact injection ENABLED — "
                "train: stochastic | "
                f"val: fixed seed={val_seed} | "
                f"test: fixed seed={test_seed}"
            )

        if self.test_local:
            train_ds = Subset(train_ds, range(32))
            val_ds   = Subset(val_ds,   range(8))
            test_ds  = Subset(test_ds,  range(8))
        elif self.max_samples is not None:
            n = min(self.max_samples, len(train_ds))
            train_ds = Subset(train_ds, range(n))
            self.logger.info(f"max_samples set: training on {n} samples")

        loader_kw = dict(
            batch_size  = self.config["batch_size"],
            num_workers = self.config.get("num_workers", 4),
            pin_memory  = True,
        )
        self.train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
        self.val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
        self.test_loader  = DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=loader_kw["num_workers"], pin_memory=True,
        )

        self.logger.info(
            f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}"
        )

        raw = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
        self._geometry = raw.geometry

    # ── Models ────────────────────────────────────────────────────────────────

    def setup_models(self):
        self.logger.info("Setting up stage models …")
        self.stages = nn.ModuleList()

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

            self.stages.append(model)
            n = sum(p.numel() for p in model.parameters())
            self.logger.info(
                f"  Stage {stage}: {n:,} parameters  (activation={activation})"
            )

        total = sum(p.numel() for m in self.stages for p in m.parameters())
        self.logger.info(f"  Total: {total:,} parameters")

        # FBP used only as a fixed pre-processor — no gradients flow through it
        self.fbp = DifferentiableParallelFBP.from_geometry(self._geometry).to(self.device)
        self.logger.info(
            f"  FBP: parallel beam | {self._geometry['num_angles']} angles | "
            f"{self._geometry['num_detectors']} detectors | "
            f"recon {self._geometry['image_size']}×{self._geometry['image_size']}"
        )

    # ── Losses ────────────────────────────────────────────────────────────────

    def setup_loss_functions(self):
        """Stage-specific loss functions (mirrors separate_trainer.py design).

        Stage 0 — Primary denoiser (UNet, sigmoid):
            Stage0Loss = 0.5 * SSIM + 0.5 * L1  on (out, gt_img)
            Drives the bulk of noise removal with balanced pixel and structural fidelity.

        Stage 1+ — Residual corrector (ResUNet, tanh):
            ResidualRefinementLoss = 0.4 * L1(residual_pred, residual_target)
                                   + 0.3 * SSIM(final_out, gt_img)
                                   + 0.3 * GradientLoss(final_out, gt_img)
            Supervises the residual directly (L1 is better than MSE for small residuals
            because it gives constant gradient direction rather than collapsing to ~0),
            while SSIM and edge-aware GradientLoss enforce structural and fine-detail
            quality on the combined output — distinct objectives from Stage 0.
        """
        self.logger.info("Setting up loss functions …")
        self.loss_fns = []
        for stage in range(self.num_stages):
            if stage == 0:
                loss_fn   = Stage0Loss().to(self.device)
                loss_desc = "Stage0Loss  (SSIM=0.5 + L1=0.5)"
            else:
                loss_fn   = ResidualRefinementLoss(
                    w_l1=0.4, w_ssim=0.3, w_grad=0.3
                ).to(self.device)
                loss_desc = "ResidualRefinementLoss  (L1_resid=0.4 + SSIM_final=0.3 + Grad_final=0.3)"
            self.loss_fns.append(loss_fn)
            self.logger.info(f"  Stage {stage}: {loss_desc}")

        self.loss_weights = self.config.get(
            "loss_weights", [1.0 / self.num_stages] * self.num_stages
        )
        for i, w in enumerate(self.loss_weights):
            self.logger.info(f"  Stage {i}: loss weight = {w:.4f}")

    # ── Optimizers ────────────────────────────────────────────────────────────

    def setup_optimizers(self):
        # Separate param group per stage so per-stage lr tuning is possible later
        param_groups = [
            {"params": m.parameters(), "lr": self.config["lr"]}
            for m in self.stages
        ]
        self.optimizer = optim.AdamW(param_groups)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.logger.info(
            f"Optimizer: AdamW | lr={self.config['lr']} per stage\n"
        )

    # ── FBP pre-processor ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _sino_to_img(self, noisy_sino: torch.Tensor) -> torch.Tensor:
        """Convert a batch of noisy sinograms to image domain via FBP.

        Applied with no_grad — FBP is a fixed pre-processor here, not part of
        the learned pipeline.  This is cheaper than the differentiable FBP used
        in lodopab_trainer.py where gradients had to flow back through the FBP.
        """
        return self.fbp(noisy_sino)   # (B, 1, H, W)

    # ── Cascade forward ───────────────────────────────────────────────────────

    def _cascade_forward(self, noisy_img: torch.Tensor) -> list[dict]:
        """Run the full cascade and return a list of per-stage info dicts.

        Stage 0 (UNet, sigmoid) — direct prediction:
            {"type": "direct", "out": stage0_out}

        Stage 1+ (ResUNet, tanh) — additive residual:
            {"type": "residual", "out": final_out,
             "residual": residual_pred, "prev_out": prev_stage_out}

        Inputs to each stage are detached so each stage's parameters receive
        gradients only from its own loss (deep supervision without cross-stage
        gradient leakage, following munet_trainer design).

        The separate "residual" and "prev_out" fields are needed by
        ResidualRefinementLoss, which supervises the residual directly
        and also evaluates the combined final output.
        """
        results = []

        # Stage 0: direct denoising
        x = self.stages[0](noisy_img)          # sigmoid → (B,1,H,W)  ∈ [0,1]
        results.append({"type": "direct", "out": x})

        # Stage 1+: additive residual correction
        for s in range(1, self.num_stages):
            prev_out = x.detach()
            residual = self.stages[s](prev_out)                        # tanh → [-1, 1]
            x        = torch.clamp(prev_out + residual, 0.0, 1.0)
            results.append({"type": "residual", "out": x,
                            "residual": residual, "prev_out": prev_out})

        return results

    # ── Gradient monitoring ───────────────────────────────────────────────────

    def _monitor_gradient_flow(self) -> dict:
        grad_info = {}
        for i, model in enumerate(self.stages):
            means, maxes = [], []
            for param in model.parameters():
                if param.grad is not None:
                    means.append(param.grad.abs().mean().item())
                    maxes.append(param.grad.abs().max().item())
            if means:
                grad_info[f"stage_{i}"] = {
                    "mean": sum(means) / len(means),
                    "max":  max(maxes),
                }
        return grad_info

    def _log_gradient_flow(self, epoch: int, batch_idx: int, grad_info: dict):
        grad_str = " | ".join(
            f"{k}: mean={v['mean']:.2e}, max={v['max']:.2e}"
            for k, v in grad_info.items()
        )
        if batch_idx % 5000 == 0:
            self.logger.info(
                f"Epoch {epoch} Batch {batch_idx} | Gradients — {grad_str}"
            )
        for k, v in grad_info.items():
            if v["mean"] < 1e-7:
                self.logger.warning(
                    f"⚠️  {k}: vanishing gradients ({v['mean']:.2e})"
                )
            elif v["mean"] > 1.0:
                self.logger.warning(
                    f"⚠️  {k}: exploding gradients ({v['mean']:.2e})"
                )

    # ── Single-epoch train ────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict:
        for m in self.stages:
            m.train()

        running = {"total": 0.0}
        for i in range(self.num_stages):
            running[f"stage{i}_loss"] = 0.0
        running_comps: dict[str, float] = {}   # stage{i}_comp_{name}

        loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch:03d}")

        for batch_idx, (noisy_sino, gt_img) in enumerate(loop):
            noisy_sino = noisy_sino.to(self.device)   # (B,1,1000,513)
            gt_img     = gt_img.to(self.device)       # (B,1,362,362)

            # Pre-process: sinogram → image domain (no grad)
            noisy_img = self._sino_to_img(noisy_sino)  # (B,1,362,362)

            self.optimizer.zero_grad()

            stage_results = self._cascade_forward(noisy_img)

            # Per-stage losses — signature differs by stage type
            per_stage_losses = []
            for i, (sr, lf, w) in enumerate(
                zip(stage_results, self.loss_fns, self.loss_weights)
            ):
                if sr["type"] == "direct":
                    # Stage 0: Stage0Loss(out, gt_img)
                    l = lf(sr["out"], gt_img) * w
                else:
                    # Stage 1+: ResidualRefinementLoss(residual, residual_target, final_out, gt_img)
                    residual_target = (gt_img - sr["prev_out"])
                    l = lf(sr["residual"], residual_target, sr["out"], gt_img) * w
                running[f"stage{i}_loss"] += l.item()
                for ck, cv in lf.components.items():
                    key = f"stage{i}_comp_{ck}"
                    running_comps[key] = running_comps.get(key, 0.0) + cv
                per_stage_losses.append(l)

            total_loss = torch.stack(per_stage_losses).sum()
            total_loss.backward()

            if batch_idx % 5000 == 0:
                grad_info = self._monitor_gradient_flow()
                self._log_gradient_flow(epoch, batch_idx, grad_info)

            nn.utils.clip_grad_norm_(
                [p for m in self.stages for p in m.parameters()],
                max_norm=1.0,
            )
            self.optimizer.step()

            running["total"] += total_loss.item()
            n_so_far = loop.n or 1
            loop.set_postfix({k: f"{v / n_so_far:.4f}" for k, v in running.items()})

        n = max(1, len(self.train_loader))
        result = {k: v / n for k, v in running.items()}
        result.update({k: v / n for k, v in running_comps.items()})
        return result

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> dict:
        for m in self.stages:
            m.eval()

        running = {"total": 0.0}
        for i in range(self.num_stages):
            running[f"stage{i}_loss"] = 0.0
            running[f"stage{i}_psnr"] = 0.0
            running[f"stage{i}_ssim"] = 0.0
        running_comps: dict[str, float] = {}   # stage{i}_comp_{name}

        for noisy_sino, gt_img in tqdm(
            self.val_loader, desc=f"Val Epoch {epoch:03d}"
        ):
            noisy_sino = noisy_sino.to(self.device)
            gt_img     = gt_img.to(self.device)

            noisy_img     = self._sino_to_img(noisy_sino)
            stage_results = self._cascade_forward(noisy_img)

            per_stage_losses = []
            for i, (sr, lf, w) in enumerate(
                zip(stage_results, self.loss_fns, self.loss_weights)
            ):
                if sr["type"] == "direct":
                    l = lf(sr["out"], gt_img) * w
                else:
                    residual_target = gt_img - sr["prev_out"]
                    l = lf(sr["residual"], residual_target, sr["out"], gt_img) * w
                per_stage_losses.append(l)
                running[f"stage{i}_loss"] += l.item()
                for ck, cv in lf.components.items():
                    key = f"stage{i}_comp_{ck}"
                    running_comps[key] = running_comps.get(key, 0.0) + cv
                # Metrics on the final image output — fixed data_range=1.0
                running[f"stage{i}_psnr"] += float(
                    compute_psnr(sr["out"], gt_img, data_range=1.0)
                )
                running[f"stage{i}_ssim"] += float(
                    compute_ssim(sr["out"], gt_img, data_range=1.0)
                )
            running["total"] += float(
                torch.stack(per_stage_losses).sum().item()
            )

        n = max(1, len(self.val_loader))
        result = {k: v / n for k, v in running.items()}
        result.update({k: v / n for k, v in running_comps.items()})
        return result

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        self.logger.info("\n" + "#" * 80)
        self.logger.info("STARTING LODOPAB DEEP SUPERVISION TRAINING")
        self.logger.info("#" * 80 + "\n")

        stopper       = EarlyStopping(patience=self.config["patience"], min_delta=0.0)
        best_val_loss = float("inf")

        for epoch in range(self.config["epochs"]):
            tr = self.train_epoch(epoch)
            va = self.validate(epoch)

            self.scheduler.step(va["total"])
            self._log_epoch_metrics(epoch, tr, va)

            self.train_losses.append(tr["total"])
            self.val_losses.append(va["total"])

            stopper.check_early_stop(va["total"])

            if va["total"] < best_val_loss:
                best_val_loss = va["total"]
                self._save_model()
                self.logger.info(f"  ✓ Best model saved at epoch {epoch}")

            if stopper.stop_training:
                self.logger.info(f"  Early stopping at epoch {epoch}")
                break

        self.logger.info("Training completed")
        self.plot_loss_curves()

    def _log_epoch_metrics(self, epoch: int, tr: dict, va: dict):
        stage_str = " | ".join(
            f"S{i} train={tr[f'stage{i}_loss']:.4f} "
            f"val={va[f'stage{i}_loss']:.4f} "
            f"PSNR={va[f'stage{i}_psnr']:.2f} "
            f"SSIM={va[f'stage{i}_ssim']:.4f}"
            for i in range(self.num_stages)
        )
        self.logger.info(
            f"Epoch {epoch:03d} | "
            f"Total train={tr['total']:.4f} val={va['total']:.4f} | "
            f"{stage_str}"
        )
        # Per-stage loss component breakdown
        for i in range(self.num_stages):
            prefix = f"stage{i}_comp_"
            comp_keys = sorted(k for k in tr if k.startswith(prefix))
            if not comp_keys:
                continue
            tr_comp = "  ".join(
                f"{k[len(prefix):]}={tr[k]:.4f}" for k in comp_keys
            )
            va_comp = "  ".join(
                f"{k[len(prefix):]}={va[k]:.4f}" for k in comp_keys
            )
            self.logger.info(f"  S{i} Train components: {tr_comp}")
            self.logger.info(f"  S{i} Val   components: {va_comp}")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_model(self):
        save_dir = os.path.join(
            self.config["checkpoints_dir"],
            self.config["model"],
            self.config["test_no"],
        )
        os.makedirs(save_dir, exist_ok=True)
        for i, m in enumerate(self.stages):
            torch.save(
                m.state_dict(),
                os.path.join(save_dir, f"stage_{i}_best.pth"),
            )

    def _load_model(self):
        base = os.path.join(
            self.config["checkpoints_dir"],
            self.config["model"],
            self.config["test_no"],
        )
        for i, m in enumerate(self.stages):
            m.load_state_dict(
                torch.load(
                    os.path.join(base, f"stage_{i}_best.pth"),
                    weights_only=True,
                )
            )

    # ── Testing ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TESTING FULL CASCADE")
        self.logger.info("=" * 80)

        self._load_model()
        for m in self.stages:
            m.eval()

        baseline_psnr = baseline_ssim = baseline_rmse = 0.0
        stage_psnr = [0.0] * self.num_stages
        stage_ssim = [0.0] * self.num_stages
        stage_rmse = [0.0] * self.num_stages

        n_test = len(self.test_loader)
        rng      = np.random.default_rng(seed=42)
        save_idx = set(
            rng.choice(n_test, size=min(10, n_test), replace=False).tolist()
        )
        self.logger.info(f"Saving visualisations for indices: {sorted(save_idx)}")

        for i, (noisy_sino, gt_img) in enumerate(
            tqdm(self.test_loader, desc="Testing")
        ):
            noisy_sino = noisy_sino.to(self.device)
            gt_img     = gt_img.to(self.device)

            noisy_img = self._sino_to_img(noisy_sino)   # (1,1,H,W)

            # Fixed data_range = 1.0 (full μ_max-normalised scale).
            # Per-image dynamic range (gt.max - gt.min) is wrong here because:
            #   1. Lung-dominant slices have a small GT range (≈0.15–0.25), making
            #      even moderate errors produce negative PSNR.
            #   2. FBP of metal-artifact sinograms outputs values far outside [0,1],
            #      whose large pixel errors are then divided by that tiny dr².
            # Using data_range=1.0 gives consistent, comparable numbers across all
            # slices and models, matching the LoDoPaB μ_max normalisation convention.
            DR = 1.0

            # Baseline: raw FBP reconstruction of noisy sinogram
            baseline_psnr += float(compute_psnr(noisy_img, gt_img, data_range=DR))
            baseline_ssim += float(compute_ssim(noisy_img, gt_img, data_range=DR))
            baseline_rmse += float(compute_rmse(noisy_img, gt_img))

            # Cascade
            stage_results = self._cascade_forward(noisy_img)
            stage_recons  = [sr["out"] for sr in stage_results]

            for s, recon in enumerate(stage_recons):
                stage_psnr[s] += float(compute_psnr(recon, gt_img, data_range=DR))
                stage_ssim[s] += float(compute_ssim(recon, gt_img, data_range=DR))
                stage_rmse[s] += float(compute_rmse(recon, gt_img))

            if i in save_idx:
                self._save_test_image(i, noisy_img, stage_recons, gt_img)

        baseline_psnr /= n_test
        baseline_ssim /= n_test
        baseline_rmse /= n_test
        stage_psnr = [v / n_test for v in stage_psnr]
        stage_ssim = [v / n_test for v in stage_ssim]
        stage_rmse = [v / n_test for v in stage_rmse]

        self.logger.info(
            f"\n[Baseline FBP(noisy)] "
            f"PSNR: {baseline_psnr:.2f} | "
            f"SSIM: {baseline_ssim:.4f} | "
            f"RMSE: {baseline_rmse:.4f}"
        )
        for s in range(self.num_stages):
            gain_psnr = stage_psnr[s] - baseline_psnr
            gain_ssim = stage_ssim[s] - baseline_ssim
            drop_rmse = baseline_rmse - stage_rmse[s]
            self.logger.info(
                f"[Stage {s}] "
                f"PSNR: {stage_psnr[s]:.2f} (+{gain_psnr:.2f}) | "
                f"SSIM: {stage_ssim[s]:.4f} (+{gain_ssim:.4f}) | "
                f"RMSE: {stage_rmse[s]:.4f} ({drop_rmse:.4f}↓)"
            )
        self.logger.info("=" * 80 + "\n")

        self.test_results = {
            "baseline": {
                "psnr": baseline_psnr,
                "ssim": baseline_ssim,
                "rmse": baseline_rmse,
            },
            "stages": {
                "psnr": stage_psnr,
                "ssim": stage_ssim,
                "rmse": stage_rmse,
            },
        }
        self._plot_stage_improvement()

    # ── Visualisation ─────────────────────────────────────────────────────────

    def _save_test_image(
        self,
        idx: int,
        noisy_img: torch.Tensor,
        stage_recons: list[torch.Tensor],
        gt_img: torch.Tensor,
    ):
        n_plots = 2 + len(stage_recons)   # FBP baseline + N stages + GT
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        vmin, vmax = 0.0, 0.45            # display window from Leuschner et al. 2021
        DR = 1.0                          # fixed data_range — full μ_max-normalised scale

        def _to_display(img_t):
            arr = img_t.cpu().squeeze().numpy()
            return np.flipud(arr.T)

        def _show(ax, img_t, title):
            ax.imshow(_to_display(img_t), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fbp_psnr = float(compute_psnr(noisy_img, gt_img, data_range=DR))
        fbp_ssim = float(compute_ssim(noisy_img, gt_img, data_range=DR))
        _show(
            axes[0], noisy_img,
            f"FBP (noisy)\nPSNR: {fbp_psnr:.2f}\nSSIM: {fbp_ssim:.4f}",
        )

        for s, recon in enumerate(stage_recons):
            s_psnr = float(compute_psnr(recon, gt_img, data_range=DR))
            s_ssim = float(compute_ssim(recon, gt_img, data_range=DR))
            _show(
                axes[s + 1], recon,
                f"Stage {s}\nPSNR: {s_psnr:.2f}\nSSIM: {s_ssim:.4f}",
            )

        _show(axes[-1], gt_img, "Ground Truth")
        plt.tight_layout()

        save_dir = os.path.join(
            self.config["output_dir"], self.config["model"],
            "fig", self.config["test_no"], "results",
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"result_{idx:04d}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    def _plot_stage_improvement(self):
        b      = self.test_results["baseline"]
        s      = self.test_results["stages"]
        labels = ["Noisy FBP"] + [f"S{i}" for i in range(self.num_stages)]
        psnr   = [b["psnr"]] + s["psnr"]
        ssim   = [b["ssim"]] + s["ssim"]
        rmse   = [b["rmse"]] + s["rmse"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, vals, ylabel in zip(
            axes, [psnr, ssim, rmse], ["PSNR", "SSIM", "RMSE"]
        ):
            ax.plot(labels, vals, marker="o")
            ax.set_title(f"{ylabel} Improvement")
            ax.set_xlabel("Stage")
            ax.set_ylabel(ylabel)
            ax.grid(True)

        plt.tight_layout()
        save_dir = os.path.join(
            self.config["output_dir"],
            self.config["model"],
            "fig",
            self.config["test_no"],
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "stage_improvement.png"), dpi=150)
        plt.close()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("TRAINING")
        self.logger.info("=" * 80)
        self.train()
        self.logger.info("=" * 80)
        self.logger.info("TESTING")
        self.logger.info("=" * 80)
        self.test()
