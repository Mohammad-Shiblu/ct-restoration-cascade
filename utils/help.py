import torch
import torch.nn as nn
import os
import torchvision
from .dataset import ImageDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from datetime import datetime
import logging

# Set up logger function
def setup_logger(save_dir):
    """Set up logging configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level= logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # logging.streamHandler()
        ]
    )
    return logging.getLogger(__name__)


# dataloader functions
def get_loaders(train_dir, train_target_dir, val_dir, val_target_dir, test_dir, test_target_dir,
                batch_size=16, train_transform=None, val_transform= None, test_transform = None, 
                num_workers=0, pin_memory=False):
    train_ds = ImageDataset(image_dir=train_dir, target_dir=train_target_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_ds = ImageDataset(image_dir=val_dir, target_dir=val_target_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    # test_ds = ImageDataset(image_dir=test_dir, target_dir=test_target_dir, transform=test_transform)
    # test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    return train_loader, val_loader

# Functions for the saving and loading the checkpoints
def save_checkpoints(model, optimizer, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_path = os.path.join(dir, "checkpoints.pth.tar")
    state_dict ={
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    print("==> Saving checkpoints..")
    torch.save(state_dict, save_path)

def load_checkpoints(model, optimizer, dir):
    print("==> Loading checkpoints...")
    load_path = os.path.join(dir, "checkpoints.pth.tar")
    checkpoint = torch.load(load_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def plot_metrics(metrics_dict, save_dir= "fig/metrics_plot.png"):
    num_metrics = len(metrics_dict)
    epochs = range(1, len(next(iter(metrics_dict.values()))) + 1)

    
    plt.figure(figsize=(5 * num_metrics, 5))

    for i, (metric_name, values) in enumerate(metrics_dict.items(), start=1):
        plt.subplot(1, num_metrics, i)
        plt.plot(epochs, values, marker='o', linestyle='-', label=metric_name)
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} per Epoch')
        plt.legend()

    plt.tight_layout()

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir)
    plt.close()
    print(f"Plot saved at {save_dir}")


