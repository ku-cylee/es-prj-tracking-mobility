import os
import torch
import numpy as np

from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, data_dir, is_train):
        data_files = os.listdir(data_dir)

        labels = list(set(self.get_label_from_filename(filename) for filename in data_files))
        self.labels = Labels(labels)

        if is_train:
            data_files = [df for idx, df in enumerate(data_files) if idx % 7 != 0]
        else:
            data_files = [df for idx, df in enumerate(data_files) if idx % 7 == 0]

        input_tensors = []
        output_tensors = []

        for filename in data_files:
            label_name = self.get_label_from_filename(filename)
            input_tensors.append(torch.FloatTensor(np.load(os.path.join(data_dir, filename))))
            output_tensors.append(self.labels.get_index(label_name))

        self.input = torch.stack(input_tensors)
        self.output = torch.LongTensor(output_tensors)


    def __len__(self):
        return len(self.output)


    def __getitem__(self, index):
        return self.input[index], self.output[index]


    def get_label_from_filename(self, filename):
        return filename.split('_')[0]


class Labels:

    def __init__(self, labels_list):
        self.idx_to_name = sorted(labels_list)
        self.name_to_idx = {label: idx for idx, label in enumerate(labels_list)}

    def get_name(self, index):
        return self.idx_to_name[index]

    def get_index(self, name):
        return self.name_to_idx[name]

    def size(self):
        return len(self.idx_to_name)
