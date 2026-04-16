import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset
import lpips

from base import BaseTrainer
from models.image_domain.unet import UNet
from utils.ct_image_dataset import CtDataset
from utils.loss import compute_loss, CombinedLoss, GrayscaleLPIPSLoss
from utils.metrics import compute_metrics, compute_ssim, compute_psnr
from utils.help import EarlyStopping


class UNetTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_data()
        self.setup_models()
        self.criterion = GrayscaleLPIPSLoss(net="vgg").to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

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
            f"Data mode: {'patch (size=' + str(patch_size) + ', patches_per_image=' + str(patches_per_image) + ')' if patch_size else 'full image (' + str(full_size) + ')'}"
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
        self.model = UNet(in_channels=1, out_channels=1).to(self.config["device"])

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self):
        early_stopping = EarlyStopping(patience=self.config["patience"], min_delta=0.0)

        self.logger.info(
            f"{self.config['test_no']}: Loss: LPIPS | Optimizer: AdamW | Scheduler: ReduceLROnPlateau"
        )

        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = 0.0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}"
            )
            for input, target in progress_bar:
                input, target = input.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, target)

                if loss.dim() > 0:
                    loss = loss.mean()
                assert loss.requires_grad

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(self.train_loader)
            val_result = self.validate()
            self.scheduler.step(val_result[0])

            self.train_losses.append(train_loss)
            self.val_losses.append(val_result[0])

            self.logger.info(
                f"Epoch {epoch+1} | Train Loss: {train_loss} | "
                f"Val Loss: {val_result[0]:.4f}, PSNR: {val_result[1]:.4f}, SSIM: {val_result[2]:.4f}"
            )

            early_stopping.check_early_stop(val_result[0])
            if early_stopping.counter == 0:
                self.save_model()
                self.logger.info(
                    f"Best model saved at epoch {epoch} with val_loss={val_result[0]}"
                )
            if early_stopping.stop_training:
                self.logger.info(f"Early stopping at epoch {epoch}")
                self.plot_loss_curves()
                break

        self.logger.info("All epochs completed.")
        self.plot_loss_curves()

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self):
        self.model.eval()
        val_loss = val_psnr = val_ssim = 0.0

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                loss = self.criterion(output, target)
                if loss.dim() > 0:
                    loss = loss.mean()
                val_loss += loss.item()
                val_psnr += compute_psnr(output, target, data_range=1.0)
                val_ssim += compute_ssim(output, target, data_range=1.0)

        n = len(self.val_loader)
        return val_loss / n, val_psnr / n, val_ssim / n

    # ── Testing ───────────────────────────────────────────────────────────────

    def test(self):
        self.load_model()
        self.logger.info("Best-val-loss model loaded for testing.")
        self.model.eval()
        ori_psnr = ori_ssim = ori_rmse = 0.0
        pred_psnr = pred_ssim = pred_rmse = 0.0

        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                original, pred = compute_metrics(input, output, target, data_range=1.0)
                ori_rmse += original[0]
                ori_psnr += original[1]
                ori_ssim += original[2]
                pred_rmse += pred[0]
                pred_psnr += pred[1]
                pred_ssim += pred[2]
                self.save_image(i, input, output, target)

        n = len(self.test_loader)
        self.logger.info(
            f"Original  PSNR: {ori_psnr/n:.4f}, SSIM: {ori_ssim/n:.4f}, RMSE: {ori_rmse/n:.4f}"
        )
        self.logger.info(
            f"Predicted PSNR: {pred_psnr/n:.4f}, SSIM: {pred_ssim/n:.4f}, RMSE: {pred_rmse/n:.4f}"
        )

    # ── Visualisation ─────────────────────────────────────────────────────────

    def save_image(self, fig_name, input, pred, target):
        input, pred, target = (
            input.cpu().numpy(),
            pred.cpu().numpy(),
            target.cpu().numpy(),
        )
        ori_metrics, pred_metrics = compute_metrics(input, pred, target)
        f, ax = plt.subplots(1, 3, figsize=(30, 10))

        ax[0].imshow(input.squeeze(), cmap="gray")
        ax[0].set_title("Noisy Image", fontsize=28)
        ax[0].set_xlabel(
            f"PSNR: {ori_metrics[1]:.3f}\nSSIM: {ori_metrics[2]:.3f}", fontsize=20
        )
        ax[1].imshow(pred.squeeze(), cmap="gray")
        ax[1].set_title("Denoised Output", fontsize=28)
        ax[1].set_xlabel(
            f"PSNR: {pred_metrics[1]:.3f}\nSSIM: {pred_metrics[2]:.3f}", fontsize=20
        )
        ax[2].imshow(target.squeeze(), cmap="gray")
        ax[2].set_title("Full Dose", fontsize=24)

        file_path = os.path.join(
            self.config["output_dir"],
            self.config["model"],
            "fig",
            self.config["test_no"],
            "results",
        )
        os.makedirs(file_path, exist_ok=True)
        f.savefig(os.path.join(file_path, f"results_{fig_name}.png"))
        plt.close()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self.logger.info("-------------------------Training------------------------")
        self.train()
        self.logger.info("-------------------------Testing------------------------")
        self.test()
