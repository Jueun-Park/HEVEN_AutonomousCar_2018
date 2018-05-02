from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control_test import Control
from lidar import Lidar
from lanecam import LaneCam
import threading
import time

platform = PlatformSerial('COM3')
lidar = Lidar()
lidar.initiate()
time.sleep(2)
lane_cam=LaneCam()
thr = threading.Thread(target=lane_cam.data_loop)
thr.start()

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