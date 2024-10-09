import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDataset(Dataset):	
    def __init__(self, image_dir, target_dir, transform = None):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images  = os.listdir(target_dir) # clean image

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        target_path = os.path.join(self.target_dir, self.images[index])         # target path (clean image)
        img_path = os.path.join(self.image_dir, f"noise_{self.images[index]}")  # input image (noisy image)
        image = Image.open(img_path).convert("L")
        target_image = Image.open(target_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            target_image = self.transform(target_image) 
            
        return image, target_image