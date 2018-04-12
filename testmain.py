from communication import PlatformSerial
from motion_planner_cuda import MotionPlanner
from car_control import Control

# port = '/dev/ttyUSB0'
port = 'COM7'
# e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.
platform = PlatformSerial(port)

motion_plan = MotionPlanner()
motion_plan.initiate()

control = Control()

while True:
    platform.recv()
    control.read(*platform.read())
    platform.write()
    platform.send()

