import numpy as np
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt

def image_selector(source_dir = "synthetic_data/imagewoof2/train/", target_dir= "synthetic_data/train/", size= 1000):
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist")
        return
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    image_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.png')):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width > 256 and height > 256:  # Only select images larger than (256, 256)
                        image_paths.append(img_path)

    selected_image = np.random.choice(image_paths, size= size, replace=False)
    for image_path in selected_image:
        file_name = os.path.basename(image_path)
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(image_path, target_path)
    print(len(image_paths))
    print(f"{len(selected_image)} iamges saved to {target_dir}")
    return image_path

def add_gaussian_noise(image_source, target_source): 
    if not os.path.exists(target_source):
        os.makedirs(target_source)
   
    for filename in os.listdir(image_source):
        img_path = os.path.join(image_source, filename)
        img = np.array(Image.open(img_path).convert('RGB')) 

        mean = np.random.uniform(0, 40)
        std = np.random.uniform(10, 50)

        noise = np.random.normal(mean, std, img.shape)
        noisy_image = img + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        noisy_image = Image.fromarray(noisy_image)
        output_path = os.path.join(target_source, f"noise_{filename}")
        noisy_image.save(output_path)


def show_image(clean_image_dir = "synthetic_data/train/clean_images/", noisy_image_dir = "synthetic_data/train/noisy_images/"):
    clean_image_list = os.listdir(clean_image_dir)
    rand_image = np.random.choice(clean_image_list)
    clean_image_path = os.path.join(clean_image_dir, rand_image)
    noisy_image_path = os.path.join(noisy_image_dir, f"noise_{rand_image}")
    clean_image = Image.open(clean_image_path)
    noise_image = Image.open(noisy_image_path)
    print(clean_image.size)
    plt.subplot(1, 2, 1)
    plt.imshow(clean_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noise_image)
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    # train image
    image_selector(source_dir="synthetic_data/imagewoof2/train/", target_dir="synthetic_data/train/clean_images/", size=1000)
    add_gaussian_noise("synthetic_data/train/clean_images/", "synthetic_data/train/noisy_images/")
    # # validation image
    # image_selector(source_dir="synthetic_data/imagewoof2/val/", target_dir="synthetic_data/val/clean_images/", size=200)
    # add_gaussian_noise("synthetic_data/val/clean_images/", "synthetic_data/val/noisy_images/")

    # show_image()
    

