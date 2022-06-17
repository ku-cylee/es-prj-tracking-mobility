import time
import numpy as np

import RPi.GPIO as GPIO

from picamera import PiCamera

from internal.car_const import PinIO

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


class Ultrasonic:

    def __init__(self, trigger, echo):
        GPIO.setmode(GPIO.BCM)

        self.trigger = trigger
        self.echo = echo

        GPIO.setup(trigger, GPIO.OUT)
        GPIO.setup(echo, GPIO.IN)
        GPIO.output(trigger, False)
        time.sleep(1)

        self.max_timeout = 300 * 2 * 29.1 * 1e-6


    def get_distance(self):
        time.sleep(.1)
        GPIO.output(self.trigger, True)
        time.sleep(1e-5)
        GPIO.output(self.trigger, False)

        pulse_start = self.get_time_after_wait(1)
        pulse_end = self.get_time_after_wait(0, pulse_start)

        distance = self.get_distance_from_duration(pulse_end - pulse_start)
        return distance


    def get_time_after_wait(self, signal, timeout=None):
        start = time.time()
        if not timeout:
            timeout = start

        while GPIO.input(self.echo) != signal:
            end = time.time()
            if (end - timeout) >= self.max_timeout:
                raise TimeoutError()

        return end


    def get_distance_from_duration(self, duration):
        # duration: us / distance: cm
        return duration * 1e6 / 2 / 29.1


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
