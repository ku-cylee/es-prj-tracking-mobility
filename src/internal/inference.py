import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def get_trained_model(model_dir):
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    model.eval()
    return model


def get_image_tensor(dir_path, filename, size=None):
    image_path = os.path.join(dir_path, filename)
    image = Image.open(image_path)
    if size:
        image = image.resize((size, size))
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


def get_centroid(output, train_size):
    ver_split, hor_split = output.shape
    centroid = torch.zeros(2)
    for vidx in range(ver_split):
        for hidx in range(hor_split):
            vcent = (vidx + .5) * train_size
            hcent = (hidx + .5) * train_size
            centroid += torch.tensor([vcent, hcent]) * output[vidx, hidx]

    centroid /= output.sum()

    return centroid


def get_normalized_centroid(output, train_size, split_count):
    centroid = get_centroid(output, train_size)
    return centroid * 2 / (train_size * split_count) - 1


def save_centroid_plot(sample_dir, sample_filename, size, dst_dir, centroid):
    sample, _ = get_image_tensor(sample_dir, sample_filename, size)
    plt.imshow(sample.transpose((1, 2, 0)), interpolation='nearest')
    plt.plot(centroid[1], centroid[0], 'bo')
    plt.savefig(os.path.join(dst_dir, f'{os.path.splitext(sample_filename)[0]}.png'))
    plt.clf()
