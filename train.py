import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from models.unet import UNet
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.help import (
    get_loaders
)

# Hyperparameter
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = False
LOAD_MODEL = False
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# dataset path 
TRAIN_INPUT_DIR = "synthetic_data/train/noisy_image/"
TRAIN_TARGET_DIR = "synthetic_data/train/images/"

VAL_INPUT_DIR = "synthetic_data/val/noisy_image/"
VAL_TARGET_DIR = "synthetic_data/val/images/"

TEST_INPUT_DIR = "synthetic_data/test/noisy_image/"
TEST_TARGET_DIR = "synthetic_data/test/images/"

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (noisy_image, clean_image) in enumerate(loop):
        noisy_image = noisy_image.to(DEVICE)
        clean_image = clean_image.to(DEVICE)

        optimizer.zero_grad()

        # forward pass
        with torch.autocast(DEVICE.type):
            outputs = model(noisy_image)
            loss = loss_fn(outputs, clean_image)

        # backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #update the tqdm loop
        loop.set_postfix(loss=loss.item())


def calculate_psnr(outputs, targets):
    mse = nn.functional.mse_loss(outputs, targets, reduction='mean').item()
    if mse == 0:  # Perfect match
        return float('inf')
    return 10 * np.log10(1 / mse)  # Assumes images are normalized between 0 and 1

# Function to calculate SSIM
def calculate_ssim(outputs, targets):
    outputs_np = outputs.squeeze().cpu().numpy()
    targets_np = targets.squeeze().cpu().numpy()
    return structural_similarity(outputs_np, targets_np, data_range=1.0)

def validate(loader, model, loss_fn):
    model.eval()
    val_loss = 0.0
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for noisy_image, clean_image in loader:
            noisy_image = noisy_image.to(DEVICE)
            clean_image = clean_image.to(DEVICE)

            outputs = model(noisy_image)
            loss = loss_fn(outputs, clean_image)
            val_loss += loss.item()

            psnr = calculate_psnr(outputs, clean_image)
            ssim = calculate_ssim(outputs, clean_image)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

    return val_loss / len(loader), np.mean(psnr_values), np.mean(ssim_values)

def main():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
    ])

    train_loader, val_loader, test_loader = get_loaders(TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, VAL_INPUT_DIR, VAL_TARGET_DIR,
                                                        TEST_INPUT_DIR, TEST_TARGET_DIR, BATCH_SIZE, transform , transform, transform)
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.GradScaler()

    for epoch in range(NUM_EPOCHS):
        model.train()
        train(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, val_psnr, val_ssim = validate(val_loader, model, loss_fn)
        print(f"Validation Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f}")


if __name__ == "__main__":
    main()
