import argparse
import numpy as np
import torch

from flask import Flask, request

from internal.inference import Centroid, get_trained_model, infer

class InferenceServer(Flask):

    def __init__(self, app_name, port, model_path, train_size, split_count):
        super().__init__(app_name)
        self.port = port
        self.model = get_trained_model(model_path)
        self.train_size = train_size
        self.split_count = split_count
        self.image_size = train_size * split_count
        self.route('/', methods=['POST'])(self.index)


    def index(self):
        image_list = request.json['image']
        image = torch.FloatTensor(np.array(image_list, dtype='uint8').reshape(3, self.image_size, self.image_size))
        output = infer(self.model, image, self.train_size, self.split_count)
        centroid = Centroid(output, self.train_size, self.split_count)

        return {
            'exists': centroid.exists,
            'offset': centroid.get_horizontal_normalized() if centroid.exists else None,
        }


    def run_server(self):
        return self.run(host='0.0.0.0', port=self.port, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', type=int, default=8080)
    parser.add_argument('--model', dest='model_path', required=True)
    parser.add_argument('--train_size', dest='train_size', type=int, default=32)
    parser.add_argument('--split', dest='split_count', type=int, default=4)
    args = parser.parse_args()

    server = InferenceServer(__name__, **vars(args))
    server.run_server()
