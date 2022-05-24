import os
import argparse
import numpy as np

from PIL import Image

def preprocess_image(args, filename):
    image_path = os.path.join(args.src_dir, filename)
    image = Image.open(image_path).resize((args.size, args.size))
    np_data = np.array(image, dtype='uint8').transpose(2, 0, 1)
    np.save(os.path.join(args.dst_dir, f'{filename}.npy'), np_data)


def get_image_list(args):
    target_images = os.listdir(args.src_dir)
    if not args.is_rewrite:
        dst_images = [filename.replace('.npy', '') for filename in os.listdir(args.dst_dir)]
        target_images = [f for f in target_images if f not in dst_images]
    return target_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src_dir', default='./data/raw')
    parser.add_argument('--dst', dest='dst_dir', default='./data/converted')
    parser.add_argument('--size', dest='size', type=int, default=32)
    parser.add_argument('--rewrite', dest='is_rewrite', type=bool, default=False) # To be fixed
    args = parser.parse_args()

    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)

    preprocess_image_filenames = get_image_list(args)
    count = len(preprocess_image_filenames)
    print(f'Preprocessing {count} images START')
    for idx, filename in enumerate(preprocess_image_filenames):
        print(f'  Processing {filename}: ({100 * (idx + 1) / count}%)')
        preprocess_image(args, filename)
    print(f'Preprocessing {count} images END')
else:
    raise ImportError('This module is not for import')
