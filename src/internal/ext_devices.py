import time
import torch
import numpy as np

import RPi.GPIO as GPIO

from picamera import PiCamera

from internal.car_const import PinIO, UltrasonicPin

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


class Ultrasonic:

    def __init__(self):
        GPIO.setup(UltrasonicPin.TRIGGER, GPIO.OUT)
        GPIO.setup(UltrasonicPin.ECHO, GPIO.IN)
        GPIO.output(UltrasonicPin.TRIGGER, False)
        time.sleep(1)


class Wheel:

    def __init__(self, en, in1, in2):
        self.en = en
        self.in1 = in1
        self.in2 = in2

        GPIO.setup(en, GPIO.OUT)
        GPIO.setup(in1, GPIO.OUT)
        GPIO.setup(in2, GPIO.OUT)
        self.pwm = GPIO.PWM(en, 100)
        self.pwm.start(0)


    def change_direction(self, offset):
        # offset is positive if car direction and wheel side is same
        # offset is negative if car direction and wheel side is opposite
        # To be implemented
        speed = offset
        return self.move_forward(speed)

    def move_forward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, PinIO.HIGH)
        GPIO.output(self.in2, PinIO.LOW)


    def stop(self):
        # Should 0 be assigned for stop?
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.in1, PinIO.LOW)
        GPIO.output(self.in2, PinIO.LOW)
