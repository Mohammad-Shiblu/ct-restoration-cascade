import numpy as np
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
from skimage.util import random_noise

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

    np.random.shuffle(image_paths)
    selected_image = np.random.choice(image_paths, size= size, replace=False)
    for image_path in selected_image:
        file_name = os.path.basename(image_path)
        target_path = os.path.join(target_dir, file_name)
        shutil.copy(image_path, target_path)
    print(len(image_paths))
    print(f"{len(selected_image)} iamges saved to {target_dir}")
    return image_path


# function for generating blcak and white patches in the image
def add_patches(image, num_patches = 4, size_range = (60, 65)):
    patched_image = image.copy()
    height, width, channel = patched_image.shape
    num = np.random.randint(3, num_patches)
    for _ in range(num):
        patch_height = np.random.randint(size_range[0], size_range[1])
        patch_width = np.random.randint(size_range[0], size_range[1])

        top_left_y = np.random.randint(0, height -patch_height)
        top_left_x = np.random.randint(0, width - patch_width)

        if np.random.rand() > 0.5:
            patch_color = (0, 0 ,0)
        else:
            patch_color = (255, 255, 255)
        
        patched_image[top_left_y: top_left_y + patch_height, top_left_x: top_left_x + patch_width] = patch_color

    return patched_image

def add_noise(image, mean = 0, var = 0.5):
    noisy_image  = random_noise(image, mode='gaussian', mean = mean, var=var, clip= True) 
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image

def generate_noisy_dataset(image_source, target_source): 
    if not os.path.exists(target_source):
        os.makedirs(target_source)
    
    for filename in os.listdir(image_source):
        img_path = os.path.join(image_source, filename)
        img = np.array(Image.open(img_path).convert('RGB'))  # img

        
        noised_image = add_noise(img)
        patched_image = add_patches(noised_image)

        noised_image = Image.fromarray(patched_image)
        output_path = os.path.join(target_source, f"noise_{filename}")
        noised_image.save(output_path)

def plot_image(clean_image_dir = "synthetic_data/train/clean_images/"):
    clean_image_list = os.listdir(clean_image_dir)
    np.random.seed(42)
    rand_image = np.random.choice(clean_image_list)
    clean_image_path = os.path.join(clean_image_dir, rand_image)
    clean_image = Image.open(clean_image_path)
    clean_image = np.array(clean_image)
    patched_image = add_patches(clean_image)
    noised_image = add_noise(patched_image)
    plt.subplot(1, 2, 1)
    plt.imshow(clean_image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(noised_image)
    plt.axis('off')
    plt.show()


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
    # image_selector(source_dir="synthetic_data/imagewoof2/val/", target_dir="synthetic_data/val/clean_images/", size=200)

    #add_noise("synthetic_data/train/clean_images/", "synthetic_data/train/noisy_images/")
    
    # plot_image()
    # # validation image
    # image_selector(source_dir="synthetic_data/imagewoof2/val/", target_dir="synthetic_data/val/clean_images/", size=200)

    generate_noisy_dataset("synthetic_data/test/clean_images/", "synthetic_data/test/noisy_images/")




    

