import argparse

from internal.car_ctrl import CarController

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', dest='train_size', type=int, default=32)
    parser.add_argument('--split', dest='split_count', type=int, default=4)
    parser.add_argument('--server_host', dest='server_host', default='localhost')
    parser.add_argument('--server_port', dest='server_port', default=8080)
    args = parser.parse_args()

    controller = CarController(**vars(args))
    controller.run_forever()
else:
    raise ImportError('This module is not for import')
