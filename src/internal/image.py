import os
import torch
import numpy as np

from PIL import Image

def get_image_list(src_dir, dst_dir, is_rewrite):
    target_images = os.listdir(src_dir)
    if not is_rewrite:
        dst_images = [filename.replace('.npy', '') for filename in os.listdir(dst_dir)]
        target_images = [f for f in target_images if f not in dst_images]
    return target_images


def get_image_np(image_path, size):
    image = Image.open(image_path).resize((size, size))
    return np.array(image, dtype='uint8').transpose(2, 0, 1)


def get_image_tensor(image_path, size):
    image_np = get_image_np(image_path, size)
    return torch.FloatTensor(image_np)


def save_image_np(src_dir, dst_dir, size, filename):
    image_path = os.path.join(src_dir, filename)
    image_np = get_image_np(image_path, size)
    return np.save(os.path.join(dst_dir, f'{filename}.npy'), image_np)
