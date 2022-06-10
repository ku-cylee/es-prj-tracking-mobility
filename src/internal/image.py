import os
import numpy as np

from PIL import Image

def get_image_list(src_dir, dst_dir, is_rewrite):
    target_images = os.listdir(src_dir)
    if not is_rewrite:
        dst_images = [filename.replace('.npy', '') for filename in os.listdir(dst_dir)]
        target_images = [f for f in target_images if f not in dst_images]
    return target_images


def preprocess_image(src_dir, dst_dir, size, filename):
    image_path = os.path.join(src_dir, filename)
    image = Image.open(image_path).resize((size, size))
    np_data = np.array(image, dtype='uint8').transpose(2, 0, 1)
    np.save(os.path.join(dst_dir, f'{filename}.npy'), np_data)
