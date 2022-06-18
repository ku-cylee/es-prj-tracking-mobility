import numpy as np
import RPi.GPIO as GPIO

from picamera import PiCamera

class Camera(PiCamera):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.resolution = (image_size, image_size)
        self.framerate = 24


    def capture_np(self):
        data = np.empty((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.capture(data, 'rgb')
        return data.transpose(2, 0, 1)


class Wheel:

    def __init__(self, en, in1, in2):
        GPIO.setmode(GPIO.BCM)

        self.en = en
        self.in1 = in1
        self.in2 = in2

        GPIO.setup(en, GPIO.OUT)
        GPIO.setup(in1, GPIO.OUT)
        GPIO.setup(in2, GPIO.OUT)
        self.pwm = GPIO.PWM(en, 100)
        self.pwm.start(0)


    def set_speed_from_offset(self, offset):
        # offset is positive if car direction and wheel side is same
        # offset is negative if car direction and wheel side is opposite
        speed = 4 * offset + 30
        return self.move_forward(speed)


    def move_forward(self, speed):
        self.pwm.ChangeDutyCycle(speed)
        GPIO.output(self.in1, 1)
        GPIO.output(self.in2, 0)


    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        GPIO.output(self.in1, 0)
        GPIO.output(self.in2, 0)
