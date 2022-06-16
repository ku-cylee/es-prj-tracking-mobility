import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from internal.lib import mkdir_nonexist
from internal.model import IdentityResNet
from internal.dataset import ImageDataset

class ImageModelTrainer:

    def __init__(self, args):
        print('Trainer Initializing...')

        self.start_time = datetime.datetime.now()
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.epoch_size = args.epoch_size
        self.data_dir = args.data_dir
        self.is_export = args.is_export
        self.export_dir = args.export_dir

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'  Device: {self.device}')

        self.trainloader, self.testloader = self.get_dataset()

        self.net = IdentityResNet(image_size=32, labels_count=2)
        self.model = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=args.learning_rate,
                                   momentum=args.momentum)


    def get_dataset(self):
        trainset = ImageDataset(self.data_dir, is_train=True)
        trainloader = DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True)

        testset = ImageDataset(self.data_dir, is_train=False)
        testloader = DataLoader(testset, batch_size=self.test_batch_size, shuffle=True)

        self.labels = trainset.labels

        return trainloader, testloader


    def train(self):
        print('Trainer Training...')
        for epoch_idx in range(self.epoch_size):
            print(f'  Running Epoch {epoch_idx + 1}/{self.epoch_size}...')

            report_loss = 0
            report_started_at = time.time()
            REPORT_PERIOD = 100
            for batch_idx, batch_data in enumerate(self.trainloader):
                local_loss = self.batchwise_train(batch_data)
                report_loss += local_loss

                if (batch_idx + 1) % 100 and batch_idx + 1 < len(self.trainloader):
                    continue

                current_time = time.time()
                avg_loss = report_loss / REPORT_PERIOD
                elapsed_time = current_time - report_started_at
                print(f'    Batch {epoch_idx + 1}.{batch_idx + 1}: Loss {avg_loss:.6f}, Elapsed Time {elapsed_time:.6f}(s)')
                report_loss = 0
                report_started_at = current_time

        print('Trainer Training END')


    def batchwise_train(self, batch):
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()


    def evaluate(self):
        print('Trainer Evaluating...')
        correct_counts = [0 for _ in range(self.labels.size())]
        total_counts = [0 for _ in range(self.labels.size())]

        with torch.no_grad():
            for batch in self.testloader:
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.net(inputs)
                predicts = (torch.max(outputs, 1)[1] == labels).squeeze()
                for label, predict in zip(labels, predicts):
                    correct_counts[label] += predict.item()
                    total_counts[label] += 1
        print('  Evaluation Report')

        print(f'    Overall Accurracy: {100 * sum(correct_counts) / sum(total_counts)}%')
        for label_idx in range(self.labels.size()):
            correct = correct_counts[label_idx]
            total = total_counts[label_idx]
            percentage = 100 * correct / total
            print(f'    {self.labels.get_name(label_idx)}: {percentage:.4f}% ({correct}/{total})')
        print('Training Evaluating END')


    def export(self):
        mkdir_nonexist(self.export_dir)

        filename = f'model-{self.start_time.strftime("%y%m%d-%H%M%S")}.pt'
        torch.save(self.model, os.path.join(self.export_dir, filename))


    def run(self):
        self.train()
        self.evaluate()
        if self.is_export:
            self.export()
