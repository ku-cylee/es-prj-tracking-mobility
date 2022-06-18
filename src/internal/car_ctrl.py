import requests

from internal.ext_devices import Camera, Wheel

class CarController:

    def __init__(self, train_size, split_count, server_host, server_port):
        self.inference_url = f'http://{server_host}:{server_port}/'

        self.camera = Camera(train_size * split_count)
        self.left_wheel = Wheel(0, 5, 6)
        self.right_wheel = Wheel(26, 13, 19)


    def stop(self):
        self.left_wheel.stop()
        self.right_wheel.stop()


    def change_direction(self, offset):
        self.left_wheel.set_speed_from_offset(-offset)
        self.right_wheel.set_speed_from_offset(2 * offset)


    def run(self):
        image = self.camera.capture_np()
        resp = requests.post(self.inference_url, json={'image': image.tolist()})
        centroid = resp.json()

        if not centroid['exists']:
            return self.stop()

        self.change_direction(centroid['offset'])


    def run_forever(self):
        while True:
            self.run()
