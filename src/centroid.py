import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def get_trained_model(model_dir):
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    model.eval()
    return model


def get_image_tensor(dir_path, filename, size):
    image_path = os.path.join(dir_path, filename)
    image = Image.open(image_path).resize((size, size))
    image_np = np.array(image, dtype='uint8').transpose(2, 0, 1)
    return image_np, torch.FloatTensor(image_np)


def infer(model, image, train_size, split_count):
    splitted_image_list = []
    for i in range(split_count):
        for j in range(split_count):
            splitted_image_list.append(image[:, i*train_size:(i+1)*train_size, j*train_size:(j+1)*train_size])

    splitted_tensor = torch.stack(splitted_image_list)
    with torch.no_grad():
        inferred = model(splitted_tensor)
        return torch.nn.Softmax()(inferred)[:,1].reshape(split_count, split_count)


def get_centroid(output, train_size, split_count, include_false=True):
    centroid = torch.zeros(2)
    for i in range(split_count):
        for j in range(split_count):
            ver_idx = (i + .5) * train_size
            hor_idx = (j + .5) * train_size
            centroid += torch.tensor([ver_idx, hor_idx]) * output[i, j]

    centroid /= output.sum()
    return centroid


def main(src_dir, dst_dir, model_dir, train_size, split_count):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    model = get_trained_model(model_dir)
    sample_filenames = os.listdir(src_dir)
    for idx, filename in enumerate(sample_filenames):
        sample_np, sample = get_image_tensor(src_dir, filename, train_size * split_count)
        print(f'Inferring {filename}: ({idx + 1}/{len(sample_filenames)})')
        output = infer(model, sample, train_size, split_count)
        centroid = get_centroid(output, train_size, split_count)
        print(f'Return: {centroid[1] * 2 / (train_size * split_count) - 1}')

        plt.imshow(sample_np.transpose((1, 2, 0)), interpolation='nearest')
        plt.plot(centroid[1], centroid[0], 'bo')
        plt.savefig(os.path.join(dst_dir, f'{os.path.splitext(filename)[0]}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src_dir', default='./data/samples')
    parser.add_argument('--dst', dest='dst_dir', default='./data/centroid-results')
    parser.add_argument('--model', dest='model_dir')
    parser.add_argument('--train_size', dest='train_size', type=int, default=32)
    parser.add_argument('--split', dest='split_count', type=int, default=4)
    args = parser.parse_args()

    main(args.src_dir, args.dst_dir, args.model_dir, args.train_size, args.split_count)
else:
    raise ImportError('This module is not for import')
