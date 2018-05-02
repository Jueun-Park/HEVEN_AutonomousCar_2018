from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control_test import Control
import time

platform = PlatformSerial('COM3')
motion = MotionPlanner()
control = Control()

while True:
    platform.recv()
    control.read(*platform.read())
    platform.status()
    platform.write(*control.write())
    platform.send()