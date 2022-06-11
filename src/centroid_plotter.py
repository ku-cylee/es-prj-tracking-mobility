import os
import argparse

from internal.inference import get_trained_model, get_image_tensor, infer, get_centroid, save_centroid_plot

def main(args):
    if not os.path.exists(args.dst_dir):
        os.mkdir(args.dst_dir)

    model = get_trained_model(args.model_dir)
    sample_filenames = os.listdir(args.src_dir)
    for idx, filename in enumerate(sample_filenames):
        image_size = args.train_size * args.split_count
        sample_np, sample = get_image_tensor(args.src_dir, filename, image_size)
        print(f'Inferring {filename}: ({idx + 1}/{len(sample_filenames)})')
        output = infer(model, sample, args.train_size, args.split_count)
        centroid = get_centroid(output, args.train_size)
        save_centroid_plot(args.src_dir, filename, image_size, args.dst_dir, centroid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src_dir', default='./data/samples')
    parser.add_argument('--dst', dest='dst_dir', default='./data/centroid-results')
    parser.add_argument('--model', dest='model_dir', required=True)
    parser.add_argument('--train_size', dest='train_size', type=int, default=32)
    parser.add_argument('--split', dest='split_count', type=int, default=4)
    args = parser.parse_args()

    main(args)
else:
    raise ImportError('This module is not for import')
