import time
import RPi.GPIO as GPIO

from internal.camera import Camera
from internal.car_const import *
from internal.inference import Centroid, get_trained_model, infer

class CarController:

    def __init__(self, model_path, train_size, split_count):
        self.model = get_trained_model(model_path)
        self.train_size = train_size
        self.split_count = split_count
        self.camera = Camera(train_size * split_count)

        GPIO.setmode(GPIO.BCM)
        self.init_ultrasonic()
        self.init_motor()


    def init_ultrasonic(self):
        GPIO.setup(UltrasonicPin.TRIGGER, GPIO.OUT)
        GPIO.setup(UltrasonicPin.ECHO, GPIO.IN)
        GPIO.output(UltrasonicPin.TRIGGER, False)
        time.sleep(1)


    def init_motor(self):
        self.pwm_left = self.set_pwm_pin(PwmPin.EN_LEFT, WheelPin.IN1_LEFT, WheelPin.IN2_LEFT)
        self.pwm_right = self.set_pwm_pin(PwmPin.EN_RIGHT, WheelPin.IN1_RIGHT, WheelPin.IN2_RIGHT)


    def set_pwm_pin(self, en, in1, in2):
        GPIO.setup(en, GPIO.OUT)
        GPIO.setup(in1, GPIO.OUT)
        GPIO.setup(in2, GPIO.OUT)
        pwm = GPIO.PWM(en, 100)
        pwm.start(0)
        return pwm


    def is_object_close(self):
        # To be implemented
        return False


    def stop(self):
        # Should 0 be assigned for stop?
        self.set_left_wheel(0, True)
        self.set_right_wheel(0, True)


    def change_direction(self, hor_offset):
        # left_speed and right_speed are defined as a function of hor_offset
        left_speed = 0
        right_speed = 0

        self.set_left_wheel(left_speed, False)
        self.set_right_wheel(right_speed, False)


    def set_left_wheel(self, speed, is_stop):
        self.pwm_left.ChangeDutyCycle(speed)
        GPIO.output(WheelPin.IN1_LEFT, PinIO.LOW if is_stop else PinIO.HIGH)
        GPIO.output(WheelPin.IN2_LEFT, PinIO.LOW)


    def set_right_wheel(self, speed, is_stop):
        self.pwm_right.ChangeDutyCycle(speed)
        GPIO.output(WheelPin.IN1_RIGHT, PinIO.LOW if is_stop else PinIO.HIGH)
        GPIO.output(WheelPin.IN2_RIGHT, PinIO.LOW)


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
