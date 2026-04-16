import torch
import torch.nn as nn
from torch.nn import AvgPool2d
import lpips
import os
import torch.nn.functional as F


def make_loss_fns(logger, num_stages, *, alpha=0.84, beta=0.16, device='cuda', use_l1_in_combined=True):

    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")

    losses = []
    losses.append(nn.L1Loss().to(device)) # first stage loss
    logger.info(f"Stage 0 loss: {losses[-1].__class__.__name__}")
    
    if num_stages >= 2:
        # Middle stages (if any): L1
        for i in range(1, num_stages):
            loss = GrayscaleLPIPSLoss(net='vgg').to(device)
            losses.append(loss)
            logger.info(f"Stage {i} loss: {losses[-1].__class__.__name__}")
        # Final stage: CombinedLoss
        # losses.append(CombinedLoss(alpha=alpha, beta=beta, l1_loss=use_l1_in_combined))
    return losses

def log_loss_scales(logger, losses, pred, target):
    """
    Helper function to log the scale of each loss for debugging
    Args:
        logger: Logger instance
        losses: List of loss functions
        pred: Predicted tensor
        target: Target tensor
    """
    logger.info("=" * 50)
    logger.info("Loss Scales Analysis:")
    logger.info("=" * 50)
    
    with torch.no_grad():
        for i, loss_fn in enumerate(losses):
            loss_val = loss_fn(pred[i], target)
            logger.info(f"Stage {i} ({loss_fn.__class__.__name__}): {loss_val.item():.6f}")
    
    logger.info("=" * 50)

# mean square loss function using mean square loss and the penalty (Not imrproved the output than normal )
def compute_loss(pred, target):
    mse_loss = F.mse_loss(pred, target)
    range_penalty = torch.mean(F.relu(pred -1.0) + F.relu(-pred))
    return mse_loss + 0.05 * range_penalty

class CombinedLoss(nn.Module):
    """
    Combined loss function for CT denoising
    Combines SSIM loss with L1/L2 loss for better convergence
    """
    def __init__(self, alpha=0.7, beta=0.3, l1_loss=True):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for SSIM loss
        self.beta = beta    # Weight for pixel-wise loss
        self.ssim_loss = SSIMLoss()
        
        if l1_loss:
            self.pixel_loss = nn.L1Loss()
        else:
            self.pixel_loss = nn.MSELoss()
            
    def forward(self, pred, target):
        ssim_loss_val = self.ssim_loss(pred, target)
        pixel_loss_val = self.pixel_loss(pred, target)
        
        total_loss = self.alpha * ssim_loss_val + self.beta * pixel_loss_val
        return total_loss

# spectrum loss
class SpectrumLoss(nn.Module):
    def forward(self, pred, target):
        # norm='ortho' divides by sqrt(N*M), keeping magnitudes ~comparable to pixel-domain losses
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        return F.mse_loss(pred_fft.real, target_fft.real) + \
               F.mse_loss(pred_fft.imag, target_fft.imag)

# calculate LPIPS loss for grayscale image in [0, 1]
class GrayscaleLPIPSLoss(nn.Module):
    def __init__(self, net= "alex"):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        for param in self.lpips.parameters():
            param.requires_grad = False

    @staticmethod
    def prep(x):
        # if gray scale repeat channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        # [0, 1] -> [-1, 1]
        return x * 2.0 - 1.0  
    
    def forward(self, pred, target):
        pred = self.prep(pred.clamp(0.0, 1.0))
        target = self.prep(target.clamp(0.0, 1.0))
        
        d = self.lpips(pred, target)
        return d.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, k1=0.01, k2=0.03, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.k1 = k1
        self.k2 = k2

        self.window = self.create_window(window_size, sigma, channel)

    def create_window(self, window_size, sigma, channel):
        """Create Gaussian window """
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2

        # 1D gaussian kernel
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        # Create 2D Gaussian kernel
        kernel = g.unsqueeze(1) * g.unsqueeze(0) # outer product
        kernel = kernel.unsqueeze(0).unsqueeze(1) # (1, 1, H, W)

        return kernel
    
    def gaussian_filter(self, input, window):
        padding = self.window_size // 2
        input_padded = F.pad(input, (padding, padding, padding, padding), mode='reflect')
        out= F.conv2d(input_padded, window, groups=self.channel)
        return out

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        if img1.device != window.device:
            window = window.to(img1.device)
        mu1 = self.gaussian_filter(img1, window)
        mu2 = self.gaussian_filter(img2, window)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.gaussian_filter(img1 * img1, window) - mu1_sq
        sigma2_sq = self.gaussian_filter(img2 * img2, window) -mu2_sq
        sigma12 = self.gaussian_filter(img1 * img2, window) - mu1_mu2

        C1 = (self.k1) ** 2
        C2 = (self.k2) ** 2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)  # Add epsilon for numerical stability
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
        
    def forward(self, img1, img2):
        """
        Forward pass - returns SSIM loss (1 - SSIM)
        Args:
            img1: Predicted/denoised image [B, C, H, W]
            img2: Ground truth/clean image [B, C, H, W]
        Returns:
            SSIM loss value (lower is better)
        """
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self.create_window(self.window_size, 1.5, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
            
        ssim_value = self.ssim(img1, img2, window, self.window_size, channel, size_average=True)
        return 1 - ssim_value  # Return loss (1 - SSIM)
    


class Stage0Loss(nn.Module):
    """
    Loss for Stage 0 (primary denoiser).

    Loss = 0.5 * SSIM + 0.5 * L1

    Weights chosen from component-scale analysis (test_no_46):
    - SpectrumLoss removed: it collapsed to 0.0000 after epoch 1, providing
      zero gradient for the rest of training. Its 0.2 weight was wasted.
    - L1 raised from 0.2 → 0.5: SSIM alone causes regression-to-mean
      over-correction (white patches become dark spots). Equal L1 weight
      brakes this asymmetry and improves pixel-level accuracy for inner
      structure details which SSIM's windowed metric misses.
    - SSIM lowered from 0.6 → 0.5: still dominant for perceptual quality
      but no longer the sole driver.
    """
    def __init__(self, w_ssim=0.5, w_l1=0.5):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_l1   = w_l1
        self.ssim   = SSIMLoss()
        self.l1     = nn.L1Loss()

    def forward(self, pred, target):
        l_ssim = self.w_ssim * self.ssim(pred, target)
        l_l1   = self.w_l1   * self.l1(pred, target)
        self.components = {
            'ssim': l_ssim.item(),
            'l1':   l_l1.item(),
        }
        return l_ssim + l_l1


class GradientLoss(nn.Module):
    """
    Sobel-based gradient loss: penalises edge/contrast mismatches between
    prediction and target in a pixel-accurate, differentiable way.

    Unlike LPIPS (which adds perceptually plausible but pixel-inaccurate texture),
    gradient loss recovers structural edges that PSNR/SSIM can also measure.
    This is why it improves both visual contrast AND PSNR/SSIM simultaneously.

    Loss = L1(Sobel_x(pred) - Sobel_x(target)) + L1(Sobel_y(pred) - Sobel_y(target))
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_gx   = F.conv2d(pred,   self.sobel_x, padding=1)
        pred_gy   = F.conv2d(pred,   self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


class ResidualRefinementLoss(nn.Module):
    """
    Loss for Stage 2 true residual learning.

    Stage 2 predicts the residual r = (target - stage1_out).
    The final output is stage1_out + r_pred.

    Loss = w_l1  * L1(r_pred, r_target)             # penalise residual error directly
         + w_ssim * SSIM_loss(final_out, target)     # structural quality of combined result
         + w_grad * GradientLoss(final_out, target)  # edge/contrast recovery (pixel-accurate)

    L1 replaces MSE on the residual because:
    - MSE residual collapses to ~4e-6 after epoch 1, providing essentially zero gradient signal
    - L1 gives constant gradient direction for small residuals → sustains useful signal throughout training

    GradientLoss replaces LPIPS because:
    - LPIPS adds perceptually plausible but pixel-inaccurate texture → PSNR/SSIM drop
    - GradientLoss recovers structural edges/contrast in a pixel-accurate way
      → both visual quality AND PSNR/SSIM improve simultaneously

    Args:
        w_l1   : weight for L1 on residual     (default 0.4)
        w_ssim : weight for SSIM on final      (default 0.3)
        w_grad : weight for gradient on final  (default 0.3)
    """
    def __init__(self, w_l1=0.4, w_ssim=0.3, w_grad=0.3):
        super().__init__()
        self.w_l1   = w_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.l1     = nn.L1Loss()
        self.ssim   = SSIMLoss()
        self.grad   = GradientLoss()

    def forward(self, residual_pred, residual_target, final_out, target):
        l_l1   = self.w_l1   * self.l1(residual_pred, residual_target)
        l_ssim = self.w_ssim * self.ssim(final_out, target)
        l_grad = self.w_grad * self.grad(final_out, target)
        # Store weighted components so the training loop can log scale differences
        self.components = {
            'l1_resid': l_l1.item(),
            'ssim_final': l_ssim.item(),
            'grad_final': l_grad.item(),
        }
        return l_l1 + l_ssim + l_grad


if __name__ == '__main__':
    pred = torch.rand(2, 1, 32, 32, requires_grad=True)  # Random predictions
    target = torch.rand(2, 1, 32, 32)
    criterion = GrayscaleLPIPSLoss(net="vgg")
    loss = criterion(pred, target)
    loss.backward()

    print("Predictions:", pred.data)
    print("Gradients: ", pred.grad)  # value should be non zero everywhere