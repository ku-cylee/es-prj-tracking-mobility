from internal.car_const import *
from internal.ext_devices import Camera, Ultrasonic, Wheel
from internal.inference import Centroid, get_trained_model, infer

class CarController:

    def __init__(self, model_path, train_size, split_count):
        self.model = get_trained_model(model_path)
        self.train_size = train_size
        self.split_count = split_count
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


    def change_direction(self, centroid):
        offset = centroid.get_horizontal_normalized()
        self.left_wheel.set_speed_from_offset(-offset)
        self.right_wheel.set_speed_from_offset(offset)


    def run(self):
        if self.is_object_close():
            return self.stop()

        image = self.camera.capture_tensor()
        output = infer(self.model, image, self.train_size, self.split_count)
        centroid = Centroid(output, self.train_size, self.split_count)

        if not centroid.exists:
            return self.stop()

        self.change_direction(centroid)


    def run_forever(self):
        while True:
            self.run()
