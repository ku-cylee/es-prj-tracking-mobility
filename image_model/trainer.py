import argparse
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from model import IdentityResNet
from dataset import ImageDataset

class ImageModelTrainer:

    def __init__(self, args):
        print('Trainer Initializing...')

        self.train_batch_size = 8
        self.test_batch_size = 4
        self.epoch_size = 50
        self.data_dir = './data/converted'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.trainloader, self.testloader = self.get_dataset()

        self.net = IdentityResNet(image_size=64, labels_count=self.labels.size())
        self.model = self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-2, momentum=.9)


    def get_dataset(self):
        transform = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
        ])

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

                if (batch_idx + 1) % 100:
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
        pass


    def run(self):
        self.train()
        self.evaluate()
        self.export()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    trainer = ImageModelTrainer(args)
    trainer.run()
else:
    raise ImportError('This module is not for import')
