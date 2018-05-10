# 2018 국제대학생창작자동차대회 자율주행차 부문 성균관대학교 HEVEN 팀
# interpreted by python 3.6
# 하위 프로그램

#######################Module#####################
from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control_test import Control

from monitor import Monitor
import time
import cv2
#####################instance#####################
motion = MotionPlanner()
control = Control()
platform = PlatformSerial('COM6')

monitor = Monitor()
#################################################


while True:
    control.read(*platform.read())

    platform.status()

    motion.motion_plan(5)
    control.mission(*motion.motionparam)

    platform.write(*control.write())

    frames = motion.getFrame()
    frame = Monitor.concatenates(frames[0], frames[1], mode='v')

    monitor.show('1', frame, frames[2], frames[3], frames[5])
    monitor.show('2', Monitor.imstatus(*platform.status()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        motion.stop()
        break
