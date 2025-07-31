from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
import json


class CtDataset(Dataset):
    def __init__(self, config, transform=None, patch_size=None):
        self.config = config
        self.transform = transform
        self.patch_size = patch_size

        data_dir = config['data_choices'][config['selected_data']]
        
        input_paths = sorted(glob.glob(os.path.join(data_dir, "*input*.npy")))
        target_paths = sorted(glob.glob(os.path.join(data_dir, "*target*.npy")))

        test_id = config['test_data']
        val_id = config['val_data']

        if config['mode'] == "train":
            self.input_path_ = [i for i in input_paths if test_id not in i and val_id not in i]
            self.target_path_ = [t for t in target_paths if test_id not in t and val_id not in t]

        elif config['mode'] == "val":
            self.input_path_ = [i for i in input_paths if val_id in i]
            self.target_path_ = [t for t in target_paths if val_id in t]

        else:  # test mode
            self.input_path_ = [i for i in input_paths if test_id in i]
            self.target_path_ = [t for t in target_paths if test_id in t]

        assert len(self.input_path_) == len(self.target_path_), "Mismatch between input and target samples"

    def __len__(self):
        return len(self.input_path_)
    
    def __getitem__(self, idx):
        input_image = np.load(self.input_path_[idx])
        target_image = np.load(self.target_path_[idx])

        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        if self.patch_size is not None:
            input_image, target_image = get_patch(input_image, target_image, self.patch_size)


        return input_image, target_image
    


def get_patch(input_image, target_image, patch_size, patch_n=10, background=0.1):
    assert input_image.shape == target_image.shape, "Input and target images must have the same shape"
    patch_input_img = []
    patch_target_img = []
    h, w = input_image.shape
    
    new_h, new_w = patch_size, patch_size
    n = 0
    while n <= patch_n:
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        patch_input_img = input_image[top:top + new_h, left:left + new_w]
        patch_target_img = target_image[top:top + new_h, left:left + new_w]

        if (np.mean(patch_input_img)< background) or (np.mean(patch_target_img) < background):
            continue
        else:
            n+=1
            patch_input_img.append(patch_input_img)
            patch_target_img.append(patch_target_img)

    return np.array(patch_input_img), np.array(patch_target_img)



if __name__ == "__main__":

    with open('../config/train.json', 'r') as f:
        config = json.load(f)


