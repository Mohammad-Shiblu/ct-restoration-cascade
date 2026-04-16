import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset


class BaseTrainer:
    """
    Shared base class for image-domain and sinogram-domain trainers.

    Contains only logic that is genuinely common to both pipelines:
      - common state initialisation
      - generic DataLoader factory
      - epoch notification for datasets that support set_epoch()
      - checkpoint save / load (config-driven path)
      - loss-curve plotting
      - abstract method stubs

    Image-domain-specific helpers (setup_data, get_transform, get_dataset,
    save_image) live in the individual image_trainer classes.
    """

    def __init__(self, config: dict, logger, test_local: bool = False):
        self.config = config
        self.device = config["device"]
        self.test_local = test_local
        self.logger = logger
        self.train_losses: list = []
        self.val_losses: list = []

    # ── DataLoader factory ────────────────────────────────────────────────────

    def get_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            num_workers=self.config["num_workers"],
        )

    # ── Epoch notification ────────────────────────────────────────────────────

    def set_train_epoch(self, epoch: int):
        """Notify the training dataset of the current epoch for varied patch sampling."""
        ds = self.train_loader.dataset
        if isinstance(ds, Subset):
            ds = ds.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_model(self):
        file_path = os.path.join(
            self.config["checkpoints_dir"], self.config["model"]
        )
        os.makedirs(file_path, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(file_path, f"checkpoints_{self.config['test_no']}.pth"),
        )

    def load_model(self):
        file_path = os.path.join(
            self.config["checkpoints_dir"],
            self.config["model"],
            f"checkpoints_{self.config['test_no']}.pth",
        )
        self.model.load_state_dict(torch.load(file_path, weights_only=True))

    # ── Loss curves ───────────────────────────────────────────────────────────

    def plot_loss_curves(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, label="Training Loss")
        plt.plot(epochs, self.val_losses, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        file_path = os.path.join(
            self.config["output_dir"],
            self.config["model"],
            "fig",
            self.config["test_no"],
        )
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(
            os.path.join(file_path, f"loss_curves{self.config['test_no']}.png")
        )
        plt.close()

    # ── Abstract stubs ────────────────────────────────────────────────────────

    def setup_models(self):
        raise NotImplementedError("setup_models() must be implemented in the subclass.")

    def train(self):
        raise NotImplementedError("train() must be implemented in the subclass.")

    def validate(self):
        raise NotImplementedError("validate() must be implemented in the subclass.")

    def test(self):
        raise NotImplementedError("test() must be implemented in the subclass.")

    def run(self):
        raise NotImplementedError("run() must be implemented in the subclass.")
