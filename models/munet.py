import torch 
import torch.nn as nn
from .unet import UNet
from .normalize import Normalize
import torch.nn.functional as F


class MUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_unets=2, features=[64, 128, 256, 512, 1024]):
        super(MUNet, self).__init__()
        self.unets = nn.ModuleList()
        self.instance_norms = nn.ModuleList()  
        for i in range(num_unets):
            if i == 0:
                self.unets.append(UNet(in_channels, out_channels, features))
            else:
                self.unets.append(UNet(out_channels, out_channels, features))
            self.instance_norms.append(Normalize())

    # def forward(self, x):
    #     outputs = []
    #     for i, unet in enumerate(self.unets):
    #         x = unet(x)
    #         x = self.instance_norms[i](x) 
    #         outputs.append(x)
    #     return outputs

    def forward(self, x):
        for i, unet in enumerate(self.unets):
            x = unet(x)
            x = self.instance_norms[i](x)
        return x


    