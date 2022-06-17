import os
import torch
import matplotlib.pyplot as plt

from internal.image import get_image_np

def get_trained_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def infer(model, image, train_size, split_count):
    splitted_image_list = []
    for i in range(split_count):
        for j in range(split_count):
            splitted_image_list.append(image[:, i*train_size:(i+1)*train_size, j*train_size:(j+1)*train_size])

    splitted_tensor = torch.stack(splitted_image_list)
    with torch.no_grad():
        inferred = model(splitted_tensor)
        return torch.nn.Softmax()(inferred)[:,1].reshape(split_count, split_count)


class Centroid:

    def __init__(self, output, train_size, split_count):
        self.image_size = train_size * split_count
        self.exists = self.object_exists(output)
        if not self.exists:
            return

        centroid = torch.zeros(2)
        for vidx in range(split_count):
            for hidx in range(split_count):
                ver_position = (vidx + .5) * train_size
                hor_position = (hidx + .5) * train_size
                position = torch.tensor([ver_position, hor_position])
                centroid += position * output[vidx, hidx]

        centroid /= output.sum()

        self.horizontal = centroid[1].item()
        self.vertical = centroid[0].item()


    def object_exists(self, output):
        return output[output > .5].shape[0] > 0


    def get_horizontal_normalized(self):
        return self.horizontal * 2 / self.image_size - 1


    def save_plot(self, sample_dir, dst_dir, sample_filename):
        image_path = os.path.join(sample_dir, sample_filename)
        sample = get_image_np(image_path, self.image_size)
        plt.imshow(sample.transpose((1, 2, 0)), interpolation='nearest')
        if self.exists:
            plt.plot(self.horizontal, self.vertical, 'bo')
        plt.savefig(os.path.join(dst_dir, f'{os.path.splitext(sample_filename)[0]}.png'))
        plt.clf()
