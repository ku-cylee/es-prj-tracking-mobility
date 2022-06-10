import argparse

from internal.trainer import ImageModelTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data_dir', default='./data/converted')
    parser.add_argument('--epoch', dest='epoch_size', type=int, default=20)
    parser.add_argument('--size', dest='image_size', type=int, default=32)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-2)
    parser.add_argument('--momentum', dest='momentum', type=float, default=.9)
    parser.add_argument('--train_batch', dest='train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch', dest='test_batch_size', type=int, default=4)
    parser.add_argument('--export', dest='is_export', type=bool, default=True) # To be fixed
    parser.add_argument('--export_dir', dest='export_dir', default='./models')

    args = parser.parse_args()
    trainer = ImageModelTrainer(args)
    trainer.run()
else:
    raise ImportError('This module is not for import')
