from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control_test import Control
from lidar import Lidar
import time

platform = PlatformSerial('COM7')
lidar = Lidar()
lidar.initiate()
time.sleep(2)

motion = MotionPlanner(lidar)
motion.initiate()
control = Control()

while True:
    platform.recv()
    control.read(*platform.read())

    if motion.target_angle is not None and motion.distance is not None:
        tup = (motion.distance, motion.target_angle)
        control.mission(10, tup, None)
        print(tup, '   ', control.steer)

    platform.write(*control.write())
    platform.send()
