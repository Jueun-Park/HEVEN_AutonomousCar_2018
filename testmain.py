from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control_test import Control

from monitor import Monitor
import time

motion = MotionPlanner()
control = Control()
platform = PlatformSerial('COM4')

monitor = Monitor()

import cv2

while True:
    platform.recv()
    control.read(*platform.read())

    platform.status()

    motion.motion_plan(1)
    control.mission(*motion.motion)

    platform.write(*control.write())
    platform.send()

    frames = motion.getFrame()
    frame = Monitor.concatenates(frames[0], frames[1], mode='v')

    monitor.show('1', frame, frames[2], frames[4])
    monitor.show('2', Monitor.imstatus(*platform.status()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        motion.stop()
        break
