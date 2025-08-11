import torch
import torch.nn as nn


class NormalizeToTarget(nn.Module):
    def __init__(self):
        super(NormalizeToTarget, self).__init__()

    def forward(sef, output, target):
        target_mean = target.mean(dim=(1 , 2, 3), keepdim=True)	
        target_std = target.std(dim=(1 , 2, 3), keepdim=True)
        output_mean = output.mean(dim=(1 , 2, 3), keepdim=True)
        output_std = output.std(dim=(1 , 2, 3), keepdim=True)

        normalized_output = (output - output_mean) / (output_std + 1e-8)
        normalized_output = normalized_output * target_std + target_mean

        return normalized_output
    

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, input):
        # Compute mean across spatial dimensions (height and width) for each channel
        mean = input.mean(dim=(2, 3), keepdim=True)  # Shape: (batch_size, channels, 1, 1)
        
        # Compute min and max across spatial dimensions (height and width) for each channel
        min_val = input.amin(dim=(2, 3), keepdim=True)  # Shape: (batch_size, channels, 1, 1)
        max_val = input.amax(dim=(2, 3), keepdim=True)  # Shape: (batch_size, channels, 1, 1)
        
        # Calculate range (max - min)
        range_val = max_val - min_val  # Shape: (batch_size, channels, 1, 1)
        range_val[range_val == 0] = 1e-8  # Avoid division by zero

        # Normalize the input tensor
        normalized_input = (input - mean) / range_val
        return normalized_input

    

        