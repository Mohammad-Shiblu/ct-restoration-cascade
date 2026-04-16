import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset

from base import BaseTrainer
from models.image_domain.munet import CascadedUNet, ImprovedCascadedUNet
from utils.ct_image_dataset import CtDataset
from utils.loss import CombinedLoss, make_loss_fns, GrayscaleLPIPSLoss, SSIMLoss, SpectrumLoss
from utils.metrics import compute_metrics, compute_ssim, compute_psnr, compute_rmse
from utils.help import EarlyStopping


class CascadedUnetTrainer(BaseTrainer):
    """
    Trainer for Cascaded U-Net with adaptive loss balancing and gradient monitoring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info(
            f"{self.config['test_no']}: Cascaded Unet (features[16,32,64,128,256]) "
            f"| Stages: {self.config['num_unets']} | Full-image training "
            f"| Stage0: LPIPS(alex, lr=1e-4) | Stage1: CombinedLoss(SSIM+L1, lr=3e-4) | Optimizer: AdamW"
        )
        self.setup_data()
        self.setup_models()
        self.setup_loss_functions()
        self.setup_optimizers()

    # ── Data setup ────────────────────────────────────────────────────────────

    def get_transform(self, target_size=None):
        ops = [transforms.ToTensor()]
        if target_size is not None:
            ops.append(transforms.Resize(target_size))
        return transforms.Compose(ops)

    def get_dataset(self, transform, mode, patch_size=None, patches_per_image=None):
        return CtDataset(
            self.config,
            transform=transform,
            mode=mode,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
        )

    def setup_data(self):
        patch_size = self.config.get("patch_size", None)
        patches_per_image = self.config.get("patches_per_image", 10)
        full_size = (self.config["image_height"], self.config["image_width"])

        train_transform = self.get_transform(target_size=None if patch_size else full_size)
        val_test_transform = self.get_transform(target_size=full_size)

        self.logger.info(
            f"Data mode: {'patch (size=' + str(patch_size) + ')' if patch_size else 'full image ' + str(full_size)}"
        )

        train_dataset = self.get_dataset(
            transform=train_transform,
            mode="train",
            patch_size=patch_size,
            patches_per_image=patches_per_image if patch_size else None,
        )
        val_dataset = self.get_dataset(transform=val_test_transform, mode="val")
        test_dataset = self.get_dataset(transform=val_test_transform, mode="test")

        if self.test_local:
            train_dataset = Subset(train_dataset, range(20))
            val_dataset = Subset(val_dataset, range(2))
            test_dataset = Subset(test_dataset, range(2))

        self.train_loader = self.get_loader(train_dataset, shuffle=True)
        self.val_loader = self.get_loader(val_dataset, shuffle=False)
        self.test_loader = self.get_loader(test_dataset, shuffle=False)
        self.logger.info("--------------data loading completed----------")

    # ── Model setup ───────────────────────────────────────────────────────────

    def setup_models(self):
        self.model = CascadedUNet(
            in_channels=1,
            out_channels=1,
            num_stages=self.config["num_unets"],
            mode="plain",
            features=[16, 32, 64, 128, 256],
        ).to(self.config["device"])

        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"Created {self.config['num_unets']}-stage Cascaded U-Net "
            f"with {num_params:,} parameters"
        )

    # ── Loss setup ────────────────────────────────────────────────────────────

    def setup_loss_functions(self):
        self.loss_fns = [
            GrayscaleLPIPSLoss(net="alex").to(self.device),
            CombinedLoss(alpha=0.7, beta=0.3, l1_loss=True).to(self.device),
        ]
        self.loss_weights = [0.5, 0.5]

    # ── Optimizer setup ───────────────────────────────────────────────────────

    def setup_optimizers(self):
        self.optimizer = optim.AdamW([
            {"params": self.model.stages[0].parameters(), "lr": 1e-4},
            {"params": self.model.stages[1].parameters(), "lr": 1e-4},
        ])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )
        self.logger.info("Optimizer: AdamW | Stage 0 lr=1e-4 | Stage 1 lr=1e-4")

    # ── Gradient monitoring ───────────────────────────────────────────────────

    def monitor_gradient_flow(self):
        grad_info = {}
        grad_max = {}
        grad_count = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if "stages.0" in name:
                    stage_key = "stage_0"
                elif "stages.1" in name:
                    stage_key = "stage_1"
                else:
                    continue

                if stage_key not in grad_info:
                    grad_info[stage_key] = []
                    grad_max[stage_key] = 0.0

                grad_mean = param.grad.abs().mean().item()
                grad_info[stage_key].append(grad_mean)
                grad_max[stage_key] = max(grad_max[stage_key], param.grad.abs().max().item())

        result = {}
        for stage in grad_info:
            result[stage] = {
                "mean": sum(grad_info[stage]) / len(grad_info[stage]),
                "max": grad_max[stage],
                "num_params": len(grad_info[stage]),
            }
        return result

    def log_gradient_flow(self, epoch, batch_idx, grad_info):
        grad_str = " | ".join([
            f"{stage}: mean={info['mean']:.2e}, max={info['max']:.2e}"
            for stage, info in grad_info.items()
        ])
        if batch_idx % 5000 == 0:
            self.logger.info(f"Epoch {epoch} Batch {batch_idx} | Gradients - {grad_str}")

        for stage, info in grad_info.items():
            if info["mean"] < 1e-7:
                self.logger.warning(f"⚠️ {stage}: Vanishing gradients ({info['mean']:.2e})")
            elif info["mean"] > 1.0:
                self.logger.warning(f"⚠️ {stage}: Exploding gradients ({info['mean']:.2e})")

        if "stage_0" in grad_info and "stage_1" in grad_info:
            ratio = grad_info["stage_0"]["mean"] / (grad_info["stage_1"]["mean"] + 1e-10)
            if ratio > 100:
                self.logger.warning(f"⚠️ Stage 0 gradients {ratio:.1f}x larger than Stage 1!")
            elif ratio < 0.01:
                self.logger.warning(f"⚠️ Stage 1 gradients {1/ratio:.1f}x larger than Stage 0!")

    # ── Training loop ─────────────────────────────────────────────────────────

    def train_stage(self, epoch):
        self.model.train()
        running = {"total": 0.0}
        for i in range(len(self.loss_fns)):
            running[f"stage{i}_loss"] = 0.0

        loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch:03d}")
        for batch_idx, (input, target) in enumerate(loop):
            input, target = input.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            stage0_out = self.model.stages[0](input)
            stage1_out = self.model.stages[1](stage0_out.detach())
            stage_outputs = [stage0_out, stage1_out]

            per_stage_losses = []
            for i, (y, lf, w) in enumerate(zip(stage_outputs, self.loss_fns, self.loss_weights)):
                l = lf(y, target) * w
                running[f"stage{i}_loss"] += float(l.detach().cpu())
                per_stage_losses.append(l)
            total_loss = torch.stack(per_stage_losses).sum()
            total_loss.backward()

            if batch_idx % 5000 == 0:
                grad_info = self.monitor_gradient_flow()
                self.log_gradient_flow(epoch, batch_idx, grad_info)

            self.optimizer.step()
            running["total"] += float(total_loss.detach().cpu())
            loop.set_postfix({k: f"{v / (loop.n or 1):.4f}" for k, v in running.items()})

        n = max(1, len(self.train_loader))
        return {k: v / n for k, v in running.items()}

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        running = {"total": 0.0}
        S = len(self.loss_fns)
        for i in range(S):
            running[f"stage{i}_loss"] = 0.0
            running[f"stage{i}_psnr"] = 0.0
            running[f"stage{i}_ssim"] = 0.0

        for input, target in tqdm(self.val_loader, desc=f"Val Epoch {epoch:03d}"):
            input, target = input.to(self.device), target.to(self.device)
            outputs = self.model(input)

            per_stage_losses = []
            for i, (y, lf, w) in enumerate(zip(outputs, self.loss_fns, self.loss_weights)):
                y = torch.clamp(y, 0.0, 1.0)
                l = lf(y, target) * w
                per_stage_losses.append(l)
                running[f"stage{i}_loss"] += float(l.cpu())
                running[f"stage{i}_psnr"] += float(compute_psnr(y, target))
                running[f"stage{i}_ssim"] += float(compute_ssim(y, target))
            running["total"] += float(torch.stack(per_stage_losses).sum().cpu())

        n = max(1, len(self.val_loader))
        return {k: v / n for k, v in running.items()}

    # ── Main training ─────────────────────────────────────────────────────────

    def train(self):
        early_stopping = EarlyStopping(patience=self.config["patience"], min_delta=0.0)

        for epoch in range(self.config["epochs"]):
            self.set_train_epoch(epoch)
            tr = self.train_stage(epoch)
            va = self.validate(epoch)

            self.scheduler.step(va["total"])
            self._log_epoch_metrics(epoch, tr, va)

            self.train_losses.append(tr["total"])
            self.val_losses.append(va["total"])

            early_stopping.check_early_stop(va["total"])
            if early_stopping.counter == 0:
                self.save_model()
                self.logger.info(
                    f"Best model saved at epoch {epoch+1} with val_loss={va['total']:.4f}"
                )
            if early_stopping.stop_training:
                self.logger.info(f"Early stopping at epoch {epoch}")
                self.plot_loss_curves()
                break

        self.logger.info("Training completed")
        self.plot_loss_curves()

    def _log_epoch_metrics(self, epoch, tr, va):
        stage_metrics_str = " | ".join(
            f"Stage {i} - Train: {tr[f'stage{i}_loss']:.4f}, "
            f"Val: {va[f'stage{i}_loss']:.4f}, "
            f"PSNR: {va[f'stage{i}_psnr']:.2f}, "
            f"SSIM: {va[f'stage{i}_ssim']:.4f}"
            for i in range(self.config["num_unets"])
        )
        self.logger.info(f"Epoch {epoch:03d} | {stage_metrics_str}")
        self.logger.info(
            f"Epoch {epoch:03d} | Train loss: {tr['total']:.4f} | Val loss: {va['total']:.4f}"
        )

    # ── Testing ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self):
        self.load_model()
        self.model.eval()
        img_idx = 0

        n_test = len(self.test_loader.dataset)
        rng = torch.Generator()
        rng.manual_seed(42)
        save_indices = set(torch.randperm(n_test, generator=rng)[:10].tolist())
        self.logger.info(f"Saving images at indices: {sorted(save_indices)}")

        baseline_psnr_sum = baseline_ssim_sum = baseline_rmse_sum = 0.0
        S = None
        stage_psnr_sum = stage_ssim_sum = stage_rmse_sum = None

        for input, target in tqdm(self.test_loader, desc="Test", leave=False):
            input, target = input.to(self.device), target.to(self.device)
            outs = self.model(input)

            if S is None:
                S = len(outs)
                stage_psnr_sum = [0.0] * S
                stage_ssim_sum = [0.0] * S
                stage_rmse_sum = [0.0] * S

            for b in range(input.size(0)):
                if img_idx in save_indices:
                    self.save_image(
                        img_idx,
                        input[b:b+1],
                        [o[b:b+1] for o in outs],
                        target[b:b+1],
                    )
                img_idx += 1

            baseline_psnr_sum += float(compute_psnr(input, target))
            baseline_ssim_sum += float(compute_ssim(input, target))
            baseline_rmse_sum += float(compute_rmse(input, target))

            for i, y in enumerate(outs):
                y = torch.clamp(y, 0.0, 1.0)
                stage_psnr_sum[i] += float(compute_psnr(y, target))
                stage_ssim_sum[i] += float(compute_ssim(y, target))
                stage_rmse_sum[i] += float(compute_rmse(y, target))

        self._log_test_metrics(
            baseline_psnr_sum, baseline_ssim_sum, baseline_rmse_sum,
            stage_psnr_sum, stage_ssim_sum, stage_rmse_sum,
            len(self.test_loader),
        )

    def _log_test_metrics(
        self,
        baseline_psnr_sum, baseline_ssim_sum, baseline_rmse_sum,
        stage_psnr_sum, stage_ssim_sum, stage_rmse_sum,
        denom,
    ):
        baseline_psnr = baseline_psnr_sum / denom
        baseline_ssim = baseline_ssim_sum / denom
        baseline_rmse = baseline_rmse_sum / denom

        stage_psnr = [v / denom for v in stage_psnr_sum]
        stage_ssim = [v / denom for v in stage_ssim_sum]
        stage_rmse = [v / denom for v in stage_rmse_sum]
        S = len(stage_psnr)

        psnr_gain_vs_base = [sp - baseline_psnr for sp in stage_psnr]
        ssim_gain_vs_base = [ss - baseline_ssim for ss in stage_ssim]
        rmse_gain_vs_base = [baseline_rmse - sr for sr in stage_rmse]
        psnr_gain_vs_prev = [None] + [stage_psnr[i] - stage_psnr[i-1] for i in range(1, S)]
        ssim_gain_vs_prev = [None] + [stage_ssim[i] - stage_ssim[i-1] for i in range(1, S)]
        rmse_gain_vs_prev = [None] + [stage_rmse[i-1] - stage_rmse[i] for i in range(1, S)]

        self.logger.info(
            f"[Baseline] PSNR {baseline_psnr:.2f}, SSIM {baseline_ssim:.4f}, RMSE {baseline_rmse:.4f}"
        )
        for i in range(S):
            prev_psnr = "" if psnr_gain_vs_prev[i] is None else f", +{psnr_gain_vs_prev[i]:.2f} vs prev"
            prev_ssim = "" if ssim_gain_vs_prev[i] is None else f", +{ssim_gain_vs_prev[i]:.4f} vs prev"
            prev_rmse = "" if rmse_gain_vs_prev[i] is None else f", {rmse_gain_vs_prev[i]:.4f}↓ vs prev"
            self.logger.info(
                f"[Stage {i}] "
                f"PSNR {stage_psnr[i]:.2f} (+{psnr_gain_vs_base[i]:.2f} vs base{prev_psnr}), "
                f"SSIM {stage_ssim[i]:.4f} (+{ssim_gain_vs_base[i]:.4f} vs base{prev_ssim}), "
                f"RMSE {stage_rmse[i]:.4f} ({rmse_gain_vs_base[i]:.4f}↓ vs base{prev_rmse})"
            )

    # ── Image saving ──────────────────────────────────────────────────────────

    def save_image(self, fig_name, input, pred_list, target):
        input = input.cpu().numpy()
        target = target.cpu().numpy()
        pred_list = [p.cpu().numpy() for p in pred_list]
        num_preds = len(pred_list)
        total_plots = 2 + num_preds

        ori_metrics, _ = compute_metrics(input, pred_list[-1], target)
        f, ax = plt.subplots(1, total_plots, figsize=(8 * total_plots, 10), constrained_layout=True)

        ax[0].imshow(input.squeeze(), cmap="gray")
        ax[0].set_title("Noisy Image", fontsize=16, pad=8)
        ax[0].set_xlabel(f"PSNR: {ori_metrics[1]:.3f}\nSSIM: {ori_metrics[2]:.4f}", fontsize=13)

        for i, pred in enumerate(pred_list):
            _, pred_metrics = compute_metrics(input, pred, target)
            ax[i+1].imshow(pred.squeeze(), cmap="gray")
            ax[i+1].set_title(f"Stage {i+1}", fontsize=16, pad=8)
            ax[i+1].set_xlabel(
                f"PSNR: {pred_metrics[1]:.3f}\nSSIM: {pred_metrics[2]:.4f}", fontsize=13
            )

        ax[-1].imshow(target.squeeze(), cmap="gray")
        ax[-1].set_title("Full Dose", fontsize=16, pad=8)

        file_path = os.path.join(
            self.config["output_dir"],
            self.config["model"],
            "fig",
            self.config["test_no"],
            "results",
        )
        os.makedirs(file_path, exist_ok=True)
        f.savefig(os.path.join(file_path, f"results_{fig_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("TRAINING")
        self.logger.info("=" * 60)
        self.train()
        self.logger.info("=" * 60)
        self.logger.info("TESTING")
        self.logger.info("=" * 60)
        self.test()
