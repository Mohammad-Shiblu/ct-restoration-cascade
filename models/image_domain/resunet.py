import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ResidualDoubleConv(nn.Module):
    """
    DoubleConv block with an intra-block residual shortcut (ResNet-style).

    Standard DoubleConv path:
        x → Conv-BN-ReLU → Conv-BN → output

    Residual path adds a shortcut before the final ReLU:
        output = ReLU( Conv-BN-ReLU → Conv-BN(x)  +  shortcut(x) )

    The shortcut is a 1×1 conv + BN when in_channels ≠ out_channels,
    or a plain identity when they match.

    Why this helps Stage 1 (residual corrector)
    --------------------------------------------
    After Stage 0 primary denoising the residual is small. A plain DoubleConv
    must learn the FULL output from scratch at each block. A residual block
    instead learns what to ADD to its input — the identity path carries the
    already-good features forward unchanged, and the conv path learns only the
    correction delta. This makes optimising small residuals much easier and
    prevents gradient vanishing through deep blocks.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # 1×1 projection shortcut when channel dims differ; identity otherwise
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class ResDown(nn.Module):
    """MaxPool2d → ResidualDoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualDoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class ResUp(nn.Module):
    """Bilinear upsample → 1×1 proj → ResidualDoubleConv (with skip cat)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv    = ResidualDoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up_conv(x)
        if x.shape != skip.shape:
            x = TF.resize(x, size=skip.shape[2:])
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class ResUNet(nn.Module):
    """
    U-Net with residual (ResNet-style) blocks at every encoder and decoder level.

    Architecture
    ------------
    Identical to the standard UNet (same encoder-decoder structure, same
    inter-level skip connections) except every DoubleConv block is replaced
    with a ResidualDoubleConv block that adds an intra-block shortcut.

    The two types of skip connections serve different roles:
        • Inter-level skips  (encoder → decoder, same as standard UNet):
          carry multi-scale spatial features across the bottleneck.
        • Intra-block residual shortcuts  (new, within each block):
          preserve the identity signal so the block learns corrections only.

    output_activation parameter
    ----------------------------
    'sigmoid'  Use for Stage 0 (primary denoiser): output in [0, 1].
    None       Use for Stage 1 (residual corrector): unconstrained output.
               The residual can be negative (Stage 0 over-shot a pixel) so
               Sigmoid would wrongly prevent downward corrections.
               The caller clamps the *final* image after adding the residual:
                   final = clamp(stage0_out + residual_pred, 0, 1)
    'tanh'     Alternative for residual mode: output in [-1, 1], symmetric.

    Args:
        in_channels        : input channels (1 for grayscale CT).
        out_channels       : output channels (1 for single residual map).
        features           : list of feature sizes per encoder level.
        output_activation  : 'sigmoid' | 'tanh' | None
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
        output_activation: str = None,
    ):
        super().__init__()
        if features is None:
            features = [16, 32, 64, 128, 256]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        self.encoders.append(ResidualDoubleConv(in_channels, features[0]))
        for i in range(len(features) - 1):
            self.encoders.append(ResDown(features[i], features[i + 1]))

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for i in reversed(range(len(features) - 1)):
            self.decoders.append(ResUp(features[i + 1], features[i]))

        # ── Output ───────────────────────────────────────────────────────────
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()   # no activation — residual can be negative

    def forward(self, x):
        skip_connections = []

        # Encoder: store all but the deepest as skip connections
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.encoders[-1](x)

        # Decoder: pair each up-block with its skip (reversed order)
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        return self.act(self.out(x))


def test():
    configs = [
        # Stage 0 usage: sigmoid output, shallow
        dict(in_channels=1, out_channels=1, features=[32, 64, 128],        output_activation='sigmoid'),
        # Stage 1 usage: no activation, full depth
        dict(in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256], output_activation=None),
    ]

    for cfg in configs:
        model = ResUNet(**cfg)
        x     = torch.randn(1, cfg['in_channels'], 256, 256)
        out   = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(
            f"features={cfg['features']}  activation={cfg['output_activation']!r:8s} "
            f"| input={tuple(x.shape)} | output={tuple(out.shape)} | params={params:,}"
        )
        assert out.shape == (1, cfg['out_channels'], 256, 256)

    print("All tests passed.")


if __name__ == '__main__':
    test()
