import torch
import torch.nn as nn


class ResidualCorrectionNet(nn.Module):
    """
    Lightweight DnCNN-style residual corrector for Stage 1+.

    Architecture: Conv → N × (Conv + BN + ReLU) → Conv
    No encoder-decoder, no max-pooling, no skip connections.

    Why not a UNet for Stage 1
    --------------------------
    After Stage 0 primary denoising, the residual is small (~0.003-0.005 RMSE).
    A deep UNet with 5 encoding levels has 1.8M parameters and learns multi-scale
    features — appropriate for the full denoising task but massively over-parameterised
    for a small correction. With so many free parameters and a near-zero target, Stage 1
    learns spurious patterns and degrades Stage 0's output.

    DnCNN (Zhang et al. 2017) showed that a plain conv stack is optimal for residual
    prediction: no pooling (preserves spatial detail), no encoder-decoder bottleneck
    (residual is a pixel-level correction, not a semantic feature), fewer parameters
    (less capacity to overfit to noise in the residual target).

    Parameters
    ----------
    in_channels  : int   Input channels. 1 = Stage 0 output only.
                         Use 2 if concatenating noisy input (see solution #2).
    num_layers   : int   Number of intermediate conv+BN+ReLU blocks. Default 6.
                         Total depth = num_layers + 2 (first + last conv).
                         DnCNN paper uses 15-17; 6-8 is sufficient for small residuals.
    num_features : int   Feature channels in hidden layers. Default 64.

    Parameter count (in_channels=1, num_layers=6, num_features=64): ~246K
    Compare: Stage 0 UNet (features=[16,32,64,128,256]): 1.81M
    """

    def __init__(self, in_channels=1, num_layers=6, num_features=64):
        super().__init__()

        layers = []

        # First conv: no BN (standard DnCNN practice for first layer)
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Intermediate blocks: Conv + BN + ReLU
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Final conv: maps features → residual (1 channel, no activation)
        # No Sigmoid: the residual can be negative (correction can go either way)
        layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1, bias=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def test():
    model = ResidualCorrectionNet(in_channels=1, num_layers=6, num_features=64)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    assert out.shape == x.shape


if __name__ == "__main__":
    test()
