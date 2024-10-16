import torch
import torch.nn as nn
import os
import torchvision
from .dataset import ImageDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np

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
def save_checkpoints(model, optimizer, dir="check_points/checkpoints.pth.tar"):
    if not os.path.exists("check_points"):
        os.makedirs("check_points")
    state_dict ={
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    print("==> Saving checkpoints..")
    torch.save(state_dict, dir)

def load_checkpoints(model, optimizer):
    print("==> Loading checkpoints...")
    checkpoint = torch.load("check_points/checkpoints.pth.tar", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Functions for calculating the metrics
def calculate_psnr(outputs, targets):
    mse = nn.functional.mse_loss(outputs, targets, reduction='mean').item()
    if mse == 0: 
        return float('inf')
    return 10 * np.log10(1 / mse)  

def calculate_ssim(outputs, targets):
    outputs_np = outputs.squeeze().cpu().detach().numpy()
    targets_np = targets.squeeze().cpu().detach().numpy()
    return structural_similarity(outputs_np, targets_np, data_range=1.0, win_size=3, multichannel = True)

def calculate_lpips(outputs, targets):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type= 'vgg', normalize=True)
    return lpips(outputs, targets)

