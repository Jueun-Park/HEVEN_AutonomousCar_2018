# 2018 국제대학생창작자동차대회 자율주행차 부문 성균관대학교 HEVEN 팀
# interpreted by python 3.6
# 하위 프로그램

#######################Module#####################q
from communication import PlatformSerial
from motion_planner import MotionPlanner
from car_control import Control
from monitor import Monitor

import time
import cv2
#####################instance#####################
motion = MotionPlanner()
control = Control()
platform = PlatformSerial('COM4')

monitor = Monitor()
#################################################


while True:
    t = time.time()
    control.read(*platform.read())
    motion.plan_motion(control.get_status())
    control.mission(*motion.get_motion_parameter())
    #control.deceleration(*motion.get_sign_trigger())
    platform.write(*control.write())

    print('!!!')

    frames = motion.get_frame()
    status_temp = monitor.concatenate(monitor.immonitor(), monitor.immission(motion.mission_num, control.get_status()), mode='h')
    status = monitor.concatenate(status_temp, monitor.imstatus(*platform.status()), mode='v')
    monitor.show('frame', *frames, windows_is=motion.windows_is)
    monitor.show('status', status)

    print(time.time() - t)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        motion.stop()
        platform.stop()
        monitor.stop()
        break
