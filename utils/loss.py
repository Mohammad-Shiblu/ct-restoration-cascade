import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, k1=0.01, k2=0.03, channel=1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.k1 = k1
        self.k2 = k2
        self.window = self._make_window(window_size, sigma, channel)

    def _make_window(self, window_size, sigma, channel):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(1)
        return kernel

    def _gauss(self, x, window):
        pad = self.window_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        return F.conv2d(x, window, groups=self.channel)

    def forward(self, img1, img2):
        ch = img1.shape[1]
        if ch != self.channel or self.window.dtype != img1.dtype:
            self.window = self._make_window(self.window_size, 1.5, ch).to(img1.device, img1.dtype)
            self.channel = ch
        w = self.window.to(img1.device, img1.dtype)
        mu1, mu2 = self._gauss(img1, w), self._gauss(img2, w)
        C1, C2 = self.k1 ** 2, self.k2 ** 2
        ssim = ((2*mu1*mu2 + C1) * (2*(self._gauss(img1*img2, w) - mu1*mu2) + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (self._gauss(img1**2, w) - mu1**2 +
                                           self._gauss(img2**2, w) - mu2**2 + C2 + 1e-8))
        return 1.0 - ssim.mean()


class GradientLoss(nn.Module):
    """Sobel-based gradient loss — recovers edges/contrast in a pixel-accurate way."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        gx = F.l1_loss(F.conv2d(pred, self.sobel_x, padding=1),
                       F.conv2d(target, self.sobel_x, padding=1))
        gy = F.l1_loss(F.conv2d(pred, self.sobel_y, padding=1),
                       F.conv2d(target, self.sobel_y, padding=1))
        return gx + gy


class Stage0Loss(nn.Module):
    """SSIM + L1 loss for Stage 0 (primary denoiser). Weights: 0.5 / 0.5."""
    def __init__(self, w_ssim=0.5, w_l1=0.5):
        super().__init__()
        self.w_ssim, self.w_l1 = w_ssim, w_l1
        self.ssim = SSIMLoss()
        self.l1   = nn.L1Loss()

    def forward(self, pred, target):
        l_ssim = self.w_ssim * self.ssim(pred, target)
        l_l1   = self.w_l1   * self.l1(pred, target)
        self.components = {'ssim': l_ssim.item(), 'l1': l_l1.item()}
        return l_ssim + l_l1


class ResidualRefinementLoss(nn.Module):
    """L1(residual) + SSIM(final) + Gradient(final) for Stage 1 residual learning.

    Weights: L1_resid=0.4, SSIM_final=0.3, Grad_final=0.3.
    """
    def __init__(self, w_l1=0.4, w_ssim=0.3, w_grad=0.3):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_grad = w_l1, w_ssim, w_grad
        self.l1   = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()

    def forward(self, residual_pred, residual_target, final_out, target):
        l_l1   = self.w_l1   * self.l1(residual_pred, residual_target)
        l_ssim = self.w_ssim * self.ssim(final_out, target)
        l_grad = self.w_grad * self.grad(final_out, target)
        self.components = {
            'l1_resid':    l_l1.item(),
            'ssim_final':  l_ssim.item(),
            'grad_final':  l_grad.item(),
        }
        return l_l1 + l_ssim + l_grad
