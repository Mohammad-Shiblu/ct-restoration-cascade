import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDataset(Dataset):	
    def __init__(self, image_dir, target_dir, transform = None):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images  = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        target_path = os.path.join(self.target_dir, f"noise_{self.images[index]}")
        image = Image.open(img_path).convert("L")
        target_image = Image.open(target_path).convert("L")

        if self.transform is not None:
            transformation = self.transform(image = image, target = target_image)
            image = transformation["image"]
            target_image = transformation["target"]
            
        return image, target_image