import argparse

from internal.car_ctrl import CarController

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
