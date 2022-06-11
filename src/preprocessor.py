import os
import argparse

from internal.lib import mkdir_nonexist
from internal.image import get_image_list, save_image_np

def main(args):
    mkdir_nonexist(args.dst_dir)

    preprocess_image_filenames = get_image_list(args.src_dir, args.dst_dir, args.is_rewrite)
    count = len(preprocess_image_filenames)
    print(f'Preprocessing {count} images START')
    for idx, filename in enumerate(preprocess_image_filenames):
        print(f'  Processing {filename}: ({100 * (idx + 1) / count}%)')
        save_image_np(args.src_dir, args.dst_dir, args.size, filename)
    print(f'Preprocessing {count} images END')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src_dir', default='./data/raw')
    parser.add_argument('--dst', dest='dst_dir', default='./data/converted')
    parser.add_argument('--size', dest='size', type=int, default=32)
    parser.add_argument('--rewrite', dest='is_rewrite', type=bool, default=False) # To be fixed
    args = parser.parse_args()

    main(args)
else:
    raise ImportError('This module is not for import')
