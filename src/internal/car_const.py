class PinIO:
    OUTPUT = 1
    INPUT = 0
    HIGH = 1
    LOW = 0

class PwmPin:
    EN_LEFT = 0
    EN_RIGHT = 26

class WheelPin:
    IN1_LEFT = 6
    IN2_LEFT = 5
    IN1_RIGHT = 19
    IN2_RIGHT = 13

class UltrasonicPin:
    TRIGGER = 23
    ECHO = 24

class Ultrasonic:
    MAX_DIST = 300
    MAX_TIMEOUT = (MAX_DIST * 2 * 29.1)

WHEEL_ACCEL_CONST = 5
