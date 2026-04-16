import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from base import BaseTrainer
from models.image_domain.unet import UNet
from models.image_domain.resunet import ResUNet
from models.image_domain.munet import CascadedUNet
from utils.ct_image_dataset import CtDataset
from utils.loss import make_loss_fns, GrayscaleLPIPSLoss, SSIMLoss, CombinedLoss, ResidualRefinementLoss, GradientLoss, SpectrumLoss, Stage0Loss
from utils.metrics import compute_psnr, compute_ssim, compute_rmse
from utils.help import EarlyStopping
import segmentation_models_pytorch as smp


class ProgressiveCascadedTrainer(BaseTrainer):
    """
    Progressive Cascaded U-Net Trainer with:
    1. Non-overlapping data splits per stage
    2. Progressive noise reduction targets
    3. Different loss functions per stage
    4. Residual learning between stages
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_stages = self.config["num_unets"]

        self.logger.info(f"{'='*80}")
        self.logger.info(f"Progressive Cascaded U-Net Trainer - Test {self.config['test_no']}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Number of stages: {self.num_stages}")
        self.logger.info(f"Training strategy: Progressive noise reduction + Non-overlapping data")
        self.logger.info(f"Patch size: {self.config.get('patch_size', 128)}")
        self.logger.info(f"Patches per image: {self.config.get('patches_per_image', 10)}")
        if "max_stage0_epochs" in self.config:
            self.logger.info(
                f"Stage 0 epoch cap: {self.config['max_stage0_epochs']} (leaves residual gap for Stage 1)"
            )

        self.setup_progressive_data()
        self.setup_models()
        self.setup_stage_losses()
        self.setup_optimizers()

        self.stage_train_losses = [[] for _ in range(self.num_stages)]
        self.stage_val_losses = [[] for _ in range(self.num_stages)]
        self.stage_best_epochs = [0] * self.num_stages

    # ── Data helpers ──────────────────────────────────────────────────────────

    def get_transform(self, target_size=None):
        ops = [transforms.ToTensor()]
        if target_size is not None:
            ops.append(transforms.Resize(target_size))
        return transforms.Compose(ops)

    def setup_progressive_data(self):
        """
        Both stages train on the full dataset.

        Splitting the data between stages is counter-productive here because:
        - We only have 7 training patients — halving leaves too little per stage.
        - Stage 1 corrects Stage 0's errors; it needs to see those errors across
          the full data distribution, not a half-dataset subset.
        - All cascaded CT denoising works in the literature train every stage on
          the full dataset sequentially.

        self.stage_train_loaders[i] and self.stage_val_loaders[i] both point to
        the same shared loaders for every stage i.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Setting up data (full dataset shared across all stages)...")
        self.logger.info("="*80)

        patch_size        = self.config.get("patch_size", None)
        patches_per_image = self.config.get("patches_per_image", 10)
        full_size         = (self.config["image_height"], self.config["image_width"])

        train_transform    = self.get_transform(target_size=None if patch_size else full_size)
        val_test_transform = self.get_transform(target_size=full_size)

        data_mode = (
            f"patch (size={patch_size}, patches_per_image={patches_per_image})"
            if patch_size
            else f"full image {full_size}"
        )
        self.logger.info(f"Data mode: {data_mode}")

        train_dataset = CtDataset(
            self.config,
            transform=train_transform,
            mode="train",
            patch_size=patch_size,
            patches_per_image=patches_per_image if patch_size else None,
        )
        val_dataset = CtDataset(
            self.config, transform=val_test_transform, mode="val", patch_size=None,
        )
        test_dataset = CtDataset(
            self.config, transform=val_test_transform, mode="test", patch_size=None,
        )

        if self.test_local:
            train_dataset = Subset(train_dataset, range(20))
            val_dataset   = Subset(val_dataset, range(2))
            test_dataset  = Subset(test_dataset, range(2))

        self.full_train_dataset = train_dataset

        shared_train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )
        shared_val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        self.stage_train_loaders = [shared_train_loader] * self.num_stages
        self.stage_val_loaders   = [shared_val_loader]   * self.num_stages

        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )

        self.logger.info(f"Train: {len(train_dataset)} samples  (all stages)")
        self.logger.info(f"Val  : {len(val_dataset)} samples  (all stages)")
        self.logger.info(f"Test : {len(test_dataset)} full images")
        self.logger.info("="*80 + "\n")

    # ── Model setup ───────────────────────────────────────────────────────────

    def setup_models(self):
        self.logger.info("Setting up stage models...")
        self.models = []

        for stage in range(self.config["num_unets"]):
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
            num_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Stage {stage} model: {num_params:,} parameters")

        total = sum(p.numel() for m in self.models for p in m.parameters())
        self.logger.info(f"Total parameters: {total:,}\n")

    # ── Loss setup ────────────────────────────────────────────────────────────

    def setup_stage_losses(self):
        """
        Stage-specific loss functions grounded in literature:

        Stage 0 — Primary Denoiser (heavy lifting):
            CombinedLoss = 0.7 * SSIM + 0.3 * L1

        Stage 1 — True Residual Corrector (DnCNN-style):
            ResidualRefinementLoss = 0.5 * MSE(residual) + 0.3 * SSIM(final) + 0.2 * LPIPS(final)
        """
        self.logger.info("Setting up stage-specific loss functions...")
        self.stage_losses = []

        for stage in range(self.num_stages):
            if stage == 0:
                loss_fn   = Stage0Loss().to(self.device)
                loss_name = "SSIM(0.6) + L1(0.2) + SpectrumLoss(0.2)"
            else:
                loss_fn   = ResidualRefinementLoss(w_l1=0.4, w_ssim=0.3, w_grad=0.3).to(self.device)
                loss_name = "L1_residual(0.4) + SSIM_final(0.3) + GradientLoss_final(0.3)"

            self.stage_losses.append(loss_fn)
            self.logger.info(f"Stage {stage}: {loss_name}")

        self.logger.info("")

    # ── Optimizer setup ───────────────────────────────────────────────────────

    def setup_optimizers(self):
        self.optimizers = []
        self.schedulers = []

        for stage in range(self.num_stages):
            optimizer = optim.AdamW(self.models[stage].parameters(), lr=self.config["lr"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

    # ── Cascade preprocessing ─────────────────────────────────────────────────

    def preprocess_with_previous_stages(self, inputs, up_to_stage):
        """Process inputs through all previous stages (frozen)."""
        if up_to_stage == 0:
            return inputs

        with torch.no_grad():
            stage_input = inputs
            for stage in range(up_to_stage):
                self.models[stage].eval()
                stage_input = self.models[stage](stage_input)
                stage_input = torch.clamp(stage_input, 0.0, 1.0)

        return stage_input

    # ── Single-epoch train / validate ────────────────────────────────────────

    def train_single_epoch(self, stage_num, epoch):
        """
        Stage 0: standard supervised denoising.
        Stage 1+: true residual learning (DnCNN-style).
        """
        model     = self.models[stage_num]
        optimizer = self.optimizers[stage_num]
        criterion = self.stage_losses[stage_num]

        model.train()
        running_loss = 0.0
        running_components: dict = {}

        train_loader = self.stage_train_loaders[stage_num]
        loop = tqdm(train_loader, desc=f"Stage {stage_num} Epoch {epoch:03d}")

        for inputs, targets in loop:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            if stage_num == 0:
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            else:
                stage0_out      = self.preprocess_with_previous_stages(inputs, stage_num)
                residual_target = targets - stage0_out
                residual_pred   = model(stage0_out)
                final_out       = stage0_out + residual_pred
                final_out       = torch.clamp(final_out, 0.0, 1.0)
                loss = criterion(residual_pred, residual_target, final_out, targets)

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for k, v in criterion.components.items():
                running_components[k] = running_components.get(k, 0.0) + v
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        n = len(train_loader)
        avg_components = {k: v / n for k, v in running_components.items()}
        return running_loss / n, avg_components

    @torch.no_grad()
    def validate_single_epoch(self, stage_num, epoch):
        """Validate a single stage — mirrors the residual logic in train_single_epoch."""
        model     = self.models[stage_num]
        criterion = self.stage_losses[stage_num]

        model.eval()
        running_loss = running_psnr = running_ssim = 0.0
        running_components: dict = {}

        for inputs, targets in self.stage_val_loaders[stage_num]:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            if stage_num == 0:
                outputs  = torch.clamp(model(inputs), 0.0, 1.0)
                loss     = criterion(outputs, targets)
                eval_out = outputs
            else:
                stage0_out      = self.preprocess_with_previous_stages(inputs, stage_num)
                residual_target = targets - stage0_out
                residual_pred   = model(stage0_out)
                final_out       = torch.clamp(stage0_out + residual_pred, 0.0, 1.0)
                loss            = criterion(residual_pred, residual_target, final_out, targets)
                eval_out        = final_out

            if loss.dim() > 0:
                loss = loss.mean()

            running_loss += loss.item()
            for k, v in criterion.components.items():
                running_components[k] = running_components.get(k, 0.0) + v
            running_psnr += compute_psnr(eval_out, targets)
            running_ssim += compute_ssim(eval_out, targets)

        n = len(self.stage_val_loaders[stage_num])
        avg_components = {k: v / n for k, v in running_components.items()}
        return {
            "loss": running_loss / n,
            "components": avg_components,
            "psnr": running_psnr / n,
            "ssim": running_ssim / n,
        }

    # ── Stage training ────────────────────────────────────────────────────────

    def train_stage(self, stage_num):
        self.logger.info("\n" + "="*80)
        self.logger.info(f"Training Stage {stage_num}")
        self.logger.info("="*80)

        early_stopping = EarlyStopping(patience=self.config["patience"], min_delta=0.0)
        best_val_loss = float("inf")

        if stage_num == 0 and "max_stage0_epochs" in self.config:
            max_epochs = min(self.config["epochs"], self.config["max_stage0_epochs"])
            self.logger.info(
                f"Stage 0 epoch cap: {max_epochs} (max_stage0_epochs={self.config['max_stage0_epochs']})"
            )
        else:
            max_epochs = self.config["epochs"]

        for epoch in range(max_epochs):
            ds = self.full_train_dataset
            if isinstance(ds, Subset):
                ds = ds.dataset
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

            train_loss, train_components = self.train_single_epoch(stage_num, epoch)
            val_metrics = self.validate_single_epoch(stage_num, epoch)

            self.schedulers[stage_num].step(val_metrics["loss"])

            train_comp_str = "  ".join(f"{k}={v:.4f}" for k, v in train_components.items())
            val_comp_str   = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics["components"].items())

            self.logger.info(
                f"Stage {stage_num} Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val PSNR: {val_metrics['psnr']:.2f} | "
                f"Val SSIM: {val_metrics['ssim']:.4f}"
            )
            self.logger.info(f"  Train components: {train_comp_str}")
            self.logger.info(f"  Val   components: {val_comp_str}")

            self.stage_train_losses[stage_num].append(train_loss)
            self.stage_val_losses[stage_num].append(val_metrics["loss"])

            early_stopping.check_early_stop(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_stage_model(stage_num)
                self.stage_best_epochs[stage_num] = epoch
                self.logger.info(f"✓ Best model saved for stage {stage_num}")

            if early_stopping.stop_training:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.load_stage_model(stage_num)
        self.logger.info(f"Loaded best model for stage {stage_num}")
        self.logger.info("="*80 + "\n")

    # ── Full training ─────────────────────────────────────────────────────────

    def train(self):
        self.logger.info("\n" + "#"*80)
        self.logger.info("STARTING PROGRESSIVE CASCADED TRAINING")
        self.logger.info("#"*80 + "\n")

        for stage in range(self.num_stages):
            self.train_stage(stage)

            self.logger.info(f"\nValidating full cascade up to stage {stage}...")
            cascade_metrics = self.validate_full_cascade(up_to_stage=stage + 1)
            self.logger.info(
                f"Cascade (Stages 0-{stage}) on full validation set: "
                f"PSNR: {cascade_metrics['psnr']:.2f} | SSIM: {cascade_metrics['ssim']:.4f}"
            )

        self.logger.info("\n" + "#"*80)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("#"*80 + "\n")

        self.plot_stage_loss_curves()

    # ── Full-cascade validation ───────────────────────────────────────────────

    @torch.no_grad()
    def validate_full_cascade(self, up_to_stage=None):
        if up_to_stage is None:
            up_to_stage = self.num_stages

        for model in self.models[:up_to_stage]:
            model.eval()

        running_psnr = running_ssim = running_rmse = 0.0

        for inputs, targets in self.val_loader:
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            stage_output = torch.clamp(self.models[0](inputs), 0.0, 1.0)

            for stage in range(1, up_to_stage):
                residual_pred = self.models[stage](stage_output)
                stage_output  = torch.clamp(stage_output + residual_pred, 0.0, 1.0)

            running_psnr += compute_psnr(stage_output, targets)
            running_ssim += compute_ssim(stage_output, targets)
            running_rmse += compute_rmse(stage_output, targets)

        n = len(self.val_loader)
        return {"psnr": running_psnr / n, "ssim": running_ssim / n, "rmse": running_rmse / n}

    # ── Testing ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self):
        self.logger.info("\n" + "="*80)
        self.logger.info("TESTING FULL CASCADE")
        self.logger.info("="*80)

        for stage in range(self.num_stages):
            self.load_stage_model(stage)
            self.models[stage].eval()

        baseline_psnr = baseline_ssim = baseline_rmse = 0.0
        stage_psnr = [0.0] * self.num_stages
        stage_ssim = [0.0] * self.num_stages
        stage_rmse = [0.0] * self.num_stages
        self.logger.info("Metrics: PSNR/RMSE on body ROI (mask > 0.02) | SSIM on full image")

        total_test = len(self.test_loader)
        rng = np.random.default_rng(seed=42)
        save_indices = set(
            rng.choice(total_test, size=min(10, total_test), replace=False).tolist()
        )
        self.logger.info(f"Saving visualizations for image indices: {sorted(save_indices)}")

        img_idx = 0

        for inputs, targets in tqdm(self.test_loader, desc="Testing"):
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)

            baseline_psnr += compute_psnr(inputs, targets, use_body_mask=True)
            baseline_ssim += compute_ssim(inputs, targets)
            baseline_rmse += compute_rmse(inputs, targets, use_body_mask=True)

            stage_outputs = []
            stage_output  = torch.clamp(self.models[0](inputs), 0.0, 1.0)
            stage_outputs.append(stage_output)

            stage_psnr[0] += compute_psnr(stage_output, targets, use_body_mask=True)
            stage_ssim[0] += compute_ssim(stage_output, targets)
            stage_rmse[0] += compute_rmse(stage_output, targets, use_body_mask=True)

            for stage in range(1, self.num_stages):
                residual_pred = self.models[stage](stage_output)
                stage_output  = torch.clamp(stage_output + residual_pred, 0.0, 1.0)
                stage_outputs.append(stage_output)

                stage_psnr[stage] += compute_psnr(stage_output, targets, use_body_mask=True)
                stage_ssim[stage] += compute_ssim(stage_output, targets)
                stage_rmse[stage] += compute_rmse(stage_output, targets, use_body_mask=True)

            if img_idx in save_indices:
                self.save_cascade_image(img_idx, inputs, stage_outputs, targets)
            img_idx += 1

        n = len(self.test_loader)
        baseline_psnr /= n
        baseline_ssim /= n
        baseline_rmse /= n
        stage_psnr = [p / n for p in stage_psnr]
        stage_ssim = [s / n for s in stage_ssim]
        stage_rmse = [r / n for r in stage_rmse]

        self.logger.info(
            f"\n[Baseline] PSNR: {baseline_psnr:.2f} | SSIM: {baseline_ssim:.4f} | RMSE: {baseline_rmse:.4f}"
        )
        for stage in range(self.num_stages):
            psnr_gain = stage_psnr[stage] - baseline_psnr
            ssim_gain = stage_ssim[stage] - baseline_ssim
            rmse_gain = baseline_rmse - stage_rmse[stage]

            if stage > 0:
                psnr_gain_prev = stage_psnr[stage] - stage_psnr[stage-1]
                ssim_gain_prev = stage_ssim[stage] - stage_ssim[stage-1]
                rmse_gain_prev = stage_rmse[stage-1] - stage_rmse[stage]
                self.logger.info(
                    f"[Stage {stage}] "
                    f"PSNR: {stage_psnr[stage]:.2f} (+{psnr_gain:.2f} vs base, +{psnr_gain_prev:.2f} vs prev) | "
                    f"SSIM: {stage_ssim[stage]:.4f} (+{ssim_gain:.4f} vs base, +{ssim_gain_prev:.4f} vs prev) | "
                    f"RMSE: {stage_rmse[stage]:.4f} ({rmse_gain:.4f}↓ vs base, {rmse_gain_prev:.4f}↓ vs prev)"
                )
            else:
                self.logger.info(
                    f"[Stage {stage}] "
                    f"PSNR: {stage_psnr[stage]:.2f} (+{psnr_gain:.2f} vs base) | "
                    f"SSIM: {stage_ssim[stage]:.4f} (+{ssim_gain:.4f} vs base) | "
                    f"RMSE: {stage_rmse[stage]:.4f} ({rmse_gain:.4f}↓ vs base)"
                )

        self.logger.info("="*80 + "\n")
        self.test_results = {
            "baseline": {
                "psnr": baseline_psnr.cpu().item() if hasattr(baseline_psnr, "cpu") else float(baseline_psnr),
                "ssim": baseline_ssim.cpu().item() if hasattr(baseline_ssim, "cpu") else float(baseline_ssim),
                "rmse": baseline_rmse.cpu().item() if hasattr(baseline_rmse, "cpu") else float(baseline_rmse),
            },
            "stages": {
                "psnr": [v.cpu().item() if hasattr(v, "cpu") else float(v) for v in stage_psnr],
                "ssim": [v.cpu().item() if hasattr(v, "cpu") else float(v) for v in stage_ssim],
                "rmse": [v.cpu().item() if hasattr(v, "cpu") else float(v) for v in stage_rmse],
            },
        }
        self.plot_stage_improvement()
        self.logger.info("\nStage improvement plot has been saved")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_stage_model(self, stage_num):
        save_dir = os.path.join(
            self.config["checkpoints_dir"], self.config["model"], self.config["test_no"]
        )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.models[stage_num].state_dict(),
            os.path.join(save_dir, f"stage_{stage_num}_best.pth"),
        )

    def load_stage_model(self, stage_num):
        load_path = os.path.join(
            self.config["checkpoints_dir"],
            self.config["model"],
            self.config["test_no"],
            f"stage_{stage_num}_best.pth",
        )
        self.models[stage_num].load_state_dict(torch.load(load_path, weights_only=True))

    # ── Visualisation ─────────────────────────────────────────────────────────

    def save_cascade_image(self, img_idx, input_img, stage_outputs, target):
        num_plots = 2 + len(stage_outputs)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        axes[0].imshow(input_img.cpu().squeeze(), cmap="gray")
        axes[0].set_title("Noisy Input")
        axes[0].axis("off")

        for i, output in enumerate(stage_outputs):
            psnr = compute_psnr(output, target)
            ssim = compute_ssim(output, target)
            axes[i+1].imshow(output.cpu().squeeze(), cmap="gray")
            axes[i+1].set_title(f"Stage {i}\nPSNR: {psnr:.2f}\nSSIM: {ssim:.4f}")
            axes[i+1].axis("off")

        axes[-1].imshow(target.cpu().squeeze(), cmap="gray")
        axes[-1].set_title("Ground Truth")
        axes[-1].axis("off")

        plt.tight_layout()

        save_dir = os.path.join(
            self.config["output_dir"],
            self.config["model"],
            "fig",
            self.config["test_no"],
            "cascade_results",
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"cascade_{img_idx:03d}.png"), dpi=150)
        plt.close()

    def plot_stage_loss_curves(self):
        fig, axes = plt.subplots(1, self.num_stages, figsize=(6 * self.num_stages, 5))
        if self.num_stages == 1:
            axes = [axes]

        for stage in range(self.num_stages):
            epochs = range(1, len(self.stage_train_losses[stage]) + 1)
            axes[stage].plot(epochs, self.stage_train_losses[stage], label="Train Loss", marker="o")
            axes[stage].plot(epochs, self.stage_val_losses[stage], label="Val Loss", marker="s")
            axes[stage].axvline(self.stage_best_epochs[stage] + 1, linestyle="--", alpha=1, color="red")
            axes[stage].set_title(f"Stage {stage} Loss Curves")
            axes[stage].set_xlabel("Epoch")
            axes[stage].set_ylabel("Loss")
            axes[stage].legend()
            axes[stage].grid(True)

        plt.tight_layout()

        save_dir = os.path.join(
            self.config["output_dir"], self.config["model"], "fig", self.config["test_no"]
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "stage_loss_curves.png"), dpi=150)
        plt.close()
        self.logger.info(f"Loss curves saved to {save_dir}")

    def plot_stage_improvement(self):
        baseline = self.test_results["baseline"]
        stages   = self.test_results["stages"]

        psnr_values  = [baseline["psnr"]] + stages["psnr"]
        ssim_values  = [baseline["ssim"]] + stages["ssim"]
        rmse_values  = [baseline["rmse"]] + stages["rmse"]
        stage_labels = ["Input"] + [f"S{i}" for i in range(self.num_stages)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(stage_labels, psnr_values, marker="o")
        axes[0].set_title("PSNR Improvement Across Cascade")
        axes[0].set_xlabel("Stage")
        axes[0].set_ylabel("PSNR")
        axes[0].grid(True)

        axes[1].plot(stage_labels, ssim_values, marker="o")
        axes[1].set_title("SSIM Improvement Across Cascade")
        axes[1].set_xlabel("Stage")
        axes[1].set_ylabel("SSIM")
        axes[1].grid(True)

        axes[2].plot(stage_labels, rmse_values, marker="o")
        axes[2].set_title("RMSE Reduction Across Cascade")
        axes[2].set_xlabel("Stage")
        axes[2].set_ylabel("RMSE")
        axes[2].grid(True)

        plt.tight_layout()

        save_dir = os.path.join(
            self.config["output_dir"], self.config["model"], "fig", self.config["test_no"]
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "cascade_stage_improvement.png"), dpi=150)
        plt.close()
        self.logger.info(f"Stage improvement plot saved to {save_dir}")

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.train()
        self.test()
