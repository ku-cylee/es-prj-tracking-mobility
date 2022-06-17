import requests

from internal.car_const import *
from internal.ext_devices import Camera, Ultrasonic, Wheel

class CarController:

    def __init__(self, train_size, split_count, server_host, server_port):
        self.inference_url = f'http://{server_host}:{server_port}/'

        self.camera = Camera(train_size * split_count)
        self.left_wheel = Wheel(PwmPin.EN_LEFT, WheelPin.IN1_LEFT, WheelPin.IN2_LEFT)
        self.right_wheel = Wheel(PwmPin.EN_RIGHT, WheelPin.IN1_RIGHT, WheelPin.IN2_RIGHT)
        self.ultrasonic = Ultrasonic(UltrasonicPin.TRIGGER, UltrasonicPin.ECHO)


    def is_object_close(self):
        try:
            distance = self.ultrasonic.get_distance()
            # Specify stopping criteria
            return distance < 1e6
        except TimeoutError:
            return False


    def stop(self):
        self.left_wheel.stop()
        self.right_wheel.stop()


    def change_direction(self, offset):
        # offset = centroid.get_horizontal_normalized()
        self.left_wheel.set_speed_from_offset(-offset)
        self.right_wheel.set_speed_from_offset(offset)


    def run(self):
        if self.is_object_close():
            return self.stop()

        image = self.camera.capture_np()
        centroid = requests.post(self.inference_url, json={'image': image.tolist()}).json()

        if not centroid['exists']:
            return self.stop()

        self.change_direction(centroid['offset'])


    def run_forever(self):
        while True:
            self.run()
