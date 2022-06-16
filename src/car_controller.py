import os
import argparse

from internal.lib import mkdir_nonexist
from internal.inference import get_trained_model, get_image_tensor, infer, Centroid
from internal.car_ctrl import CarController

def main(src_dir, dst_dir, model_path, train_size, split_count):
    model = get_trained_model(model_path)
    sample_filenames = os.listdir(src_dir)
    for idx, filename in enumerate(sample_filenames):
        image_size = train_size * split_count
        _, sample = get_image_tensor(src_dir, filename, image_size)
        print(f'Inferring {filename}: ({idx + 1}/{len(sample_filenames)})')
        output = infer(model, sample, train_size, split_count)
        centroid = Centroid(output, train_size, split_count)
        centroid.save_plot(src_dir, dst_dir, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model_path', required=True)
    parser.add_argument('--train_size', dest='train_size', type=int, default=32)
    parser.add_argument('--split', dest='split_count', type=int, default=4)
    args = parser.parse_args()

    controller = CarController(**vars(args))
    controller.run_forever()
else:
    raise ImportError('This module is not for import')
