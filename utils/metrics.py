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


def body_mask(target, threshold=0.02):
    """
    Returns a boolean mask of the patient body (foreground), excluding the large
    black background common in CT images.

    CT images normalised to [0,1] have background ≈ 0. A threshold of 0.02
    (≈ -960 HU for a [-1000, 1000] window) captures the air/tissue boundary.

    Why this matters: background pixels are 0 in both prediction and target,
    contributing 0 to MSE. This artificially reduces MSE and inflates PSNR by
    4–5 dB, making results incomparable with papers that evaluate on the ROI only.

    Args:
        target : torch.Tensor  [B, C, H, W] or [C, H, W] — ground truth image
        threshold : float — pixels below this are considered background

    Returns:
        torch.BoolTensor same shape as target
    """
    return target > threshold


def compute_psnr(image, target, data_range=1.0, use_body_mask=False):
    """
    Compute PSNR. When use_body_mask=True, evaluates only on foreground
    (patient body) pixels, consistent with how most CT denoising papers report.
    """
    if use_body_mask:
        mask = body_mask(target)
        if isinstance(image, torch.Tensor):
            mse = torch.mean((image[mask] - target[mask]) ** 2)
        else:
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else (target > 0.02)
            mse = np.mean((image[mask_np] - target[mask_np]) ** 2)
    else:
        if isinstance(image, torch.Tensor):
            mse = torch.mean((image - target) ** 2)
        else:
            mse = np.mean((image - target) ** 2)

    if isinstance(mse, torch.Tensor):
        if mse == 0:
            return float('inf')
        return (10 * torch.log10((data_range ** 2) / mse)).item()
    else:
        if mse == 0:
            return float('inf')
        return 10 * np.log10((data_range ** 2) / mse)


def compute_rmse(image, target, use_body_mask=False):
    if use_body_mask:
        mask = body_mask(target)
        if isinstance(image, torch.Tensor):
            return torch.sqrt(torch.mean((image[mask] - target[mask]) ** 2))
        else:
            mask_np = target > 0.02
            return np.sqrt(np.mean((image[mask_np] - target[mask_np]) ** 2))
    if isinstance(image, torch.Tensor):
        return torch.sqrt(torch.mean((image - target) ** 2))
    else:
        return np.sqrt(np.mean((image - target) ** 2))


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

    # SSIM operates on full image (background is consistent between pred and target,
    # so it does not inflate the score the way PSNR does).
    ssim_value = ssim(image, target, data_range=data_range, size_average=False)
    return ssim_value.item() if ssim_value.ndimension() == 0 else ssim_value.mean().item()
    
    





    

    
    
    
    



    


    
    



