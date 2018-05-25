# 2018 International Student Car Competition: Autonomous Car SKKU Team. HEVEN
# 2018 국제대학생창작자동차대회 자율주행차 부문 성균관대학교 HEVEN 팀
# interpreted by python 3.6
# please read README.md

# =================Module===================
import cv2
from src.car_control import Control
from src.monitor import Monitor
from src.motion_planner import MotionPlanner
from src.communication import PlatformSerial
# =================instance=================
planner = MotionPlanner()
controller = Control()
platform = PlatformSerial('COM4')  # check your serial port
monitor = Monitor()
# ==========================================


while True:
    # perception - planning - control loop
    controller.read(*platform.read())
    planner.plan_motion(controller.get_status())
    print("mission num: ", planner.mission_num)
    print("sign mission num: ", planner.sign_cam.mission_number)
    print("key mission num: ", planner.key_cam.mission_num)
    print()
    controller.mission(*planner.get_motion_parameter())
    platform.write(*controller.write())

    # show monitor
    frames = planner.get_frame()
    status_temp = monitor.concatenate(monitor.immonitor(), monitor.immission(planner.mission_num, controller.get_status()),
                                      mode='h')
    status = monitor.concatenate(status_temp, monitor.imstatus(*platform.status()), mode='v')
    monitor.show('frame', *frames, windows_is=planner.windows_is)
    monitor.show('status', status)

    # exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        planner.stop()
        platform.stop()
        monitor.stop()
        break
