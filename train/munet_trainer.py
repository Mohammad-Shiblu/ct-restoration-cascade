import torch
import torch.nn as nn
import torch.optim as optim
from models.munet import MUNet
from train.base import BaseTrainer
from train.generic_trainer import UNetTrainer
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
import os
os.environ['TORCH_HOME'] = '/home/woody/iwi5/iwi5240h/Masters-Thesis/pretrained/'
import matplotlib.pyplot as plt

class MUnetTrainer(UNetTrainer, BaseTrainer):
    
    def setup_models(self):
        """Initialize the M-Unet model"""
        self.model = MUNet(in_channels=3, out_channels=3, num_unets = self.config['num_unet']).to(self.device)



    



        

            
        