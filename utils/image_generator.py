import numpy as np
from PIL import Image
import os

def image_generator(size= (256, 256), no_images = 500, directory= "synthetic_data/images/", mode = 'L'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(no_images):
        image_array = np.random.randint(0, 255, size, dtype=np.uint8)   # np.random.normal(mean, stddev, size).clip(0, 255).astype(np.uint8) 
        img = Image.fromarray(image_array, mode='L')
        img.save(f"{directory}/img_{i}.png")


def add_gaussian_noise(image_source, target_source): 
    if not os.path.exists(target_source):
        os.makedirs(target_source)
    mean = 0
    std = 25

    for filename in os.listdir(image_source):
        img_path = os.path.join(image_source, filename)
        img = np.array(Image.open(img_path).convert('L')) # converting the image to grayscale

        noise = np.random.normal(mean, std, img.shape)
        noisy_image = img + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        noisy_image = Image.fromarray(noisy_image)
        output_path = os.path.join(target_source, f"noise_{filename}")
        noisy_image.save(output_path)

if __name__ == '__main__':
    # image for training
    SIZE = (256, 256)

    image_generator(size=SIZE, no_images= 1000, directory="synthetic_data/train/images")
    add_gaussian_noise("synthetic_data/train/images/", "synthetic_data/train/noisy_image/")

    # Image for validation
    image_generator(size=SIZE, no_images=200, directory="synthetic_data/val/images")
    add_gaussian_noise("synthetic_data/val/images/", "synthetic_data/val/noisy_image/")

    # Image for testing
    image_generator(size=SIZE, no_images=200, directory="synthetic_data/test/images")
    add_gaussian_noise("synthetic_data/test/images/", "synthetic_data/test/noisy_image/")
