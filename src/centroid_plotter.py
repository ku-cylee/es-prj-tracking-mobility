import os
import argparse

from internal.image import get_image_tensor
from internal.lib import mkdir_nonexist
from internal.inference import get_trained_model, infer, Centroid

def main(src_dir, dst_dir, model_path, train_size, split_count):
    mkdir_nonexist(dst_dir)

    model = get_trained_model(model_path)
    sample_filenames = os.listdir(src_dir)
    for idx, filename in enumerate(sample_filenames):
        image_size = train_size * split_count
        image_path = os.path.join(src_dir, filename)
        sample = get_image_tensor(image_path, image_size)
        print(f'Inferring {filename}: ({idx + 1}/{len(sample_filenames)})')
        output = infer(model, sample, train_size, split_count)
        centroid = Centroid(output, train_size, split_count)
        centroid.save_plot(src_dir, dst_dir, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src_dir', default='./data/samples')
    parser.add_argument('--dst', dest='dst_dir', default='./data/centroid-results')
    parser.add_argument('--model', dest='model_path', required=True)
    parser.add_argument('--train_size', dest='train_size', type=int, default=32)
    parser.add_argument('--split', dest='split_count', type=int, default=4)
    args = parser.parse_args()

    main(**vars(args))
else:
    raise ImportError('This module is not for import')
