import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine = True),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, affine= True),
            nn.ReLU(inplace= True),
        )

    def forward(self, x):
        return self.conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down =  nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_c):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        if x.shape != skip_c.shape:
            x = TF.resize(x, size=skip_c.shape[2:])
        x = torch.cat((skip_c, x), dim=1)
        x = self.conv(x)
        return x
        
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512, 1024],
                 output_activation='sigmoid'):
        """
        Args:
            output_activation:
                'sigmoid' — output in [0, 1]. Use for Stage 0 (predicts clean image).
                'tanh'    — output in [-1, 1]. Use for Stage 1 (predicts residual).
                            Residual can be negative when Stage 0 over-estimated a pixel,
                            so Sigmoid would wrongly block downward corrections.
                            The caller clamps the final image after addition:
                                final = clamp(stage0_out + residual_pred, 0, 1)
                            SSIM/GradientLoss receive this clamped final — always in [0,1].
                None      — no activation (identity). Not recommended; can diverge.
        """
        super(UNet, self).__init__()
        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.encoders.append(DoubleConv(in_channels, features[0]))
        for i in range(len(features) - 1):
            self.encoders.append(Down(features[i], features[i+1]))

        # --- Decoder ---
        self.decoders = nn.ModuleList()
        for i in reversed(range(len(features) - 1)):
            self.decoders.append(Up(features[i+1], features[i]))

        # --- Output ---
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)
        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        skip_connections = []

        # Encoder: store all except the last output as skip connection
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skip_connections.append(x)

        # Bottlenext deepest encoder, no_skip saved here
        x = self.encoders[-1](x)

        # Decoder: pair each up block with its skip connection (reversed)
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        return self.act(self.out(x))

  

def test():
    configs = [
        (1, 1, [64, 128, 256, 512, 1024]),   # original depth
        (1, 1, [8, 16, 32]),                 # shallow / lightweight
        (3, 2, [32, 64, 128, 256]),          # 3-channel input, 2-class output
    ]

    for in_c, out_c, features in configs:
        x = torch.randn((1, in_c, 256, 256))
        model = UNet(in_channels=in_c, out_channels=out_c, features=features)
        preds = model(x)
        print(f"features={features} | input={tuple(x.shape)} | output={tuple(preds.shape)}")
        assert preds.shape == (1, out_c, 256, 256), "Shape mismatch!"
    print("All tests passed!")

if __name__ == "__main__":
    test()