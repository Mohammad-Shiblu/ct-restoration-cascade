from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import glob
import torch


class CtDataset(Dataset):
    def __init__(self, config, transform=None, mode='train', patch_size=None, patches_per_image=10):
        self.config = config
        self.transform = transform
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image

        data_dir = config['data_choices'][config['selected_data']]

        input_paths = sorted(glob.glob(os.path.join(data_dir, "*input*.npy")))
        target_paths = sorted(glob.glob(os.path.join(data_dir, "*target*.npy")))

        test_id = config['test_data']
        val_id = config['val_data']

        if mode == "train":
            self.input_path_ = [i for i in input_paths if test_id not in i and val_id not in i]
            self.target_path_ = [t for t in target_paths if test_id not in t and val_id not in t]

        elif mode == "val":
            self.input_path_ = [i for i in input_paths if val_id in i]
            self.target_path_ = [t for t in target_paths if val_id in t]

        elif mode == "test":
            self.input_path_ = [i for i in input_paths if test_id in i]
            self.target_path_ = [t for t in target_paths if test_id in t]
        else:
            raise ValueError("Mode must be either 'train', 'val' or 'test'.")

        assert len(self.input_path_) == len(self.target_path_), "Mismatch between input and target samples"

        self.epoch = 0

    def set_epoch(self, epoch):
        """Call at the start of each training epoch so patch positions vary."""
        self.epoch = epoch

    def __len__(self):
        # If using patches, multiply dataset size by patches_per_image
        if self.patch_size is not None:
            return len(self.input_path_) * self.patches_per_image
        return len(self.input_path_)

    def __getitem__(self, idx):
        # If using patches, map idx to (image_idx, patch_idx)
        if self.patch_size is not None:
            image_idx = idx // self.patches_per_image
            patch_idx = idx % self.patches_per_image
        else:
            image_idx = idx
            patch_idx = None

        # Load the full image (as numpy array) — cast to float32 immediately to halve
        # the in-memory buffer size (disk files are float64) and avoid a redundant
        # float64→float32 tensor copy later.
        input_image = np.load(self.input_path_[image_idx]).astype(np.float32)
        target_image = np.load(self.target_path_[image_idx]).astype(np.float32)

        # Extract a single patch if patch_size is specified (BEFORE transform)
        if self.patch_size is not None:
            input_image, target_image = get_single_patch(
                input_image, target_image, self.patch_size,
                seed=self.epoch * 1_000_000 + idx  # varies per epoch so patches differ each pass
            )

        # Apply transforms if any (AFTER patch extraction)
        if self.transform is not None:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        else:
            # If no transform, convert to tensor manually
            # Ensure correct shape: (C, H, W)
            if len(input_image.shape) == 2:
                input_image = torch.from_numpy(input_image).unsqueeze(0)  # Add channel dimension
                target_image = torch.from_numpy(target_image).unsqueeze(0)
            else:
                input_image = torch.from_numpy(input_image)
                target_image = torch.from_numpy(target_image)

        return input_image.float(), target_image.float()


def get_single_patch(input_image, target_image, patch_size, background_threshold=0.1, max_attempts=100, seed=None):
    """
    Extract a SINGLE random patch from input and target images.
    Rejects patches with mean intensity below background_threshold.

    Args:
        input_image: Input image (H, W) or (C, H, W) as numpy array
        target_image: Target image (H, W) or (C, H, W) as numpy array
        patch_size: Size of square patch
        background_threshold: Minimum mean intensity to accept patch
        max_attempts: Maximum number of attempts to find valid patch
        seed: Random seed for reproducibility

    Returns:
        Tuple of (input_patch, target_patch) as numpy arrays
    """
    assert input_image.shape == target_image.shape, "Input and target images must have the same shape"

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Handle both (H, W) and (C, H, W) shapes
    if len(input_image.shape) == 3:
        _, h, w = input_image.shape
    else:
        h, w = input_image.shape

    assert h >= patch_size and w >= patch_size, f"Image size ({h}, {w}) must be >= patch_size ({patch_size})"

    # Try to find a valid patch
    for attempt in range(max_attempts):
        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)

        if len(input_image.shape) == 3:
            patch_input = input_image[:, top:top + patch_size, left:left + patch_size]
            patch_target = target_image[:, top:top + patch_size, left:left + patch_size]
        else:
            patch_input = input_image[top:top + patch_size, left:left + patch_size]
            patch_target = target_image[top:top + patch_size, left:left + patch_size]

        # Check if patch has sufficient foreground content
        if np.mean(patch_input) >= background_threshold and np.mean(patch_target) >= background_threshold:
            return patch_input, patch_target

    # If no valid patch found after max_attempts, return the last patch anyway
    print(f"Warning: Could not find valid patch after {max_attempts} attempts. Returning last patch.")
    return patch_input, patch_target
