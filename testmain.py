from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control_test import Control

from monitor import Monitor
import time

platform = PlatformSerial('COM3')
motion = MotionPlanner()
control = Control()

import cv2
while True:
    platform.recv()
    control.read(*platform.read())

    platform.status()
    motion.loop()
    if motion.target_angle is not None and motion.distance is not None:
        control.mission(10, (motion.distance, motion.target_angle), None)

    platform.write(*control.write())
    platform.send()

    frames = motion.getFrame()
    frame = Monitor.concatenates(frames[0], frames[1], mode='v')
    Monitor.show(frame, frames[2], Monitor.imstatus(*platform.status()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        motion.stop()
        break