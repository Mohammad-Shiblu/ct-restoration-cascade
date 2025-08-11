import torch
import numpy as np
from pytorch_msssim import ssim


def compute_metrics(input, pred, target, data_range=1.0):
    input_rmse = compute_rmse(input, target)
    input_psnr = compute_psnr(input, target, data_range)
    input_ssim = compute_ssim(input, target, data_range)
    pred_rmse = compute_rmse(pred, target)
    pred_psnr = compute_psnr(pred, target, data_range)
    pred_ssim = compute_ssim(pred, target, data_range)
    return (input_rmse, input_psnr, input_ssim), (pred_rmse, pred_psnr, pred_ssim)
    


def compute_psnr(image, target, data_range=1.0):
    if isinstance(image, torch.Tensor):
        mse = torch.mean((image - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * torch.log10((data_range ** 2) / mse)
        return psnr.item()
    elif isinstance(image, np.ndarray):
        mse = np.mean((image - target) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 10 * np.log10((data_range ** 2) / mse)
        return psnr
    else:
        raise TypeError("Input images must be either torch.Tensor or numpy.ndarray")

    
def compute_rmse(image, target):
    if isinstance(image, torch.Tensor):
        return torch.sqrt(torch.mean((image-target)**2))
    else:
        return np.sqrt(np.mean((image - target)**2))
    
def compute_ssim(image, target, data_range=1):
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    if isinstance(target, np.ndarray):
        target = torch.tensor(target)
    if image.ndimension() == 3:  # [C, H, W] → [1, C, H, W]
        image = image.unsqueeze(0)
        target = target.unsqueeze(0)

    if image.ndimension() == 2:  # [H, W] → [1, 1, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)

    ssim_value = ssim(image, target, data_range=data_range, size_average=False) # this function expect a 4d or 5d array
    return ssim_value.item() if ssim_value.ndimension() == 0 else ssim_value.mean().item()
    
    





    

    
    
    
    



    


    
    



