import torch
import numpy as np

from picamera import PiCamera

class Camera(PiCamera):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.resolution = (image_size, image_size)
        self.framerate = 24


    def capture_tensor(self):
        data = np.empty((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.capture(data, 'rgb')
        return torch.FloatTensor(data.transpose(2, 0, 1))
