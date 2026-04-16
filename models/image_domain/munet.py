import torch 
import torch.nn as nn
from .unet import UNet
from .residual_unet import ImprovedUNet
import torch.nn.functional as F


# cascaded unet model with three option plain (simple in-out), concat with input, residual
class CascadedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_stages=2, mode="plain", features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.features = features
        self.mode = mode

        self.stages = nn.ModuleList()
       
        
        for i in range(num_stages):
            if i == 0:
                # First stage takes original input and the behavior is same is for all modes
                stage_in_ch = in_channels
            else:
                if mode == "plain":
                    # only previous stage output is fed 
                    stage_in_ch = out_channels
                else:
                    # 'concat' and 'residual' takes the input and previous outputs
                    stage_in_ch = in_channels + out_channels

            self.stages.append(UNet(stage_in_ch, out_channels, features))
            

    def forward(self, x):
        outputs = []
        current_input = x

        # stage 1: same for all modes
        y = self.stages[0](x)
        outputs.append(y)
        # stage 2
        for i in range(1, self.num_stages):
            if self.mode == "plain": # just use the output of the previous stage
                stage_input = y
                y = self.stages[i](stage_input)
                outputs.append(y)

            elif self.mode == 'concat':
                stage_input = torch.cat([x, y], dim=1)
                y = self.stages[i](stage_input)
                outputs.append(y)

            else:   # residual
                stage_input = torch.cat([x, y], dim=1)
                r_i = self.stages[i](stage_input)
                y = y + r_i
                # y = y - r_i  # as we are denoising, subtract the predicted noise
                outputs.append(y)

        return outputs

# improved_munet.py
class ImprovedCascadedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_stages=2, 
                 mode="concat", features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.mode = mode
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        
        for i in range(num_stages):
            if i == 0:
                stage_in = in_channels
            else:
                # Always concatenate input with previous output
                stage_in = in_channels + out_channels
            
            self.stages.append(ImprovedUNet(stage_in, out_channels, features))
    
    def forward(self, x):
        outputs = []
        y = self.stages[0](x)
        outputs.append(y)
        
        for i in range(1, self.num_stages):
            # Always use concat mode for better context
            stage_input = torch.cat([x, y], dim=1)
            
            if self.mode == "difference":
                # Predict refinement
                diff = self.stages[i](stage_input)
                y = torch.clamp(y + diff, 0.0, 1.0)
            else:
                # Direct prediction
                y = self.stages[i](stage_input)
            
            outputs.append(y)
        
        return outputs
            



    