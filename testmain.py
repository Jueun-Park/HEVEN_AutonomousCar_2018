from lane_cam import LaneCam
from lidar import Lidar
from communication import PlatformSerial
from default_test1 import Control
from serialpacket import SerialPacket

import threading
import time

lane_cam = LaneCam()

lane1 = threading.Thread(target=lane_cam.left_camera_loop)
lane2 = threading.Thread(target=lane_cam.right_camera_loop)
lane3 = threading.Thread(target=lane_cam.show_loop)

lane1.start()
lane2.start()
lane3.start()

current_lidar = Lidar()
current_lidar.initiate()

# port = '/dev/ttyUSB0'
port = 'COM7'
# e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.
platform = PlatformSerial(port)
control = Control(0, 50, 0)

while True:
    t1 = time.time()
    platform._read()
    #platform.write_packet = SerialPacket(steer=control.steer, speed=control.speed, gear=control.gear, brake=control.brake)
    platform._write(control.speed, control.steer, control.gear)
    print(control.steer, control.speed, control.gear, control.brake)
    t2 = time.time()
    print(t2 - t1)
    print()

