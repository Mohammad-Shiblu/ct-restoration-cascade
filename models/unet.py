import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine = True),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine= True),
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
        self.up = F.interpolate
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_c):
        x = self.up(x, scale_factor=2)
        x = self.up_conv(x)
        if x.shape != skip_c.shape:
            x = TF.resize(x, size=skip_c.shape[2:])
        x = torch.concat((skip_c, x), dim=1)
        x = self.conv(x)
        return x
        

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.first = DoubleConv(in_channels, features[0]) 
        self.down1 = Down(features[0], features[1])     
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[4])
        self.up4 = Up(features[4], features[3])
        self.up3 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up1 = Up(features[1], features[0])
        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.first(x)  # x1 = sk_c value for first top layer
        x2 = self.down1(x1)  # x2 = sk_c value for 2nd  layer
        x3 = self.down2(x2)  # x3 = sk_c value for 3rd top layer
        x4 = self.down3(x3)  # x4 = sk_c value for 4th layer
        x = self.down4(x4)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)
        return x

def test():
    x = torch.randn((3, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()