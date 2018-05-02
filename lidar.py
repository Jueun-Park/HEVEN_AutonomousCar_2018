# 라이다 통신 및 해석(장애물 추출)
# input: LiDAR
# output: numpy array? (to path_planner)

import math
import socket
import threading
import time
import numpy as np
import cv2

modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
         'MOVING_OBS': 3, 'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}


class Lidar:
    RADIUS = 500  # 원일 경우 반지름, 사각형일 경우 한 변
    NOISE_THRESHOLD = 3000  # 노이즈 분류 기준

    def __init__(self):
        self.HOST = '169.254.248.220'
        self.PORT = 2111
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.data_list = None

        self.frame = None

        self.stop_fg = False

        self.sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lidar.connect((self.HOST, self.PORT))
        self.sock_lidar.send(str.encode(self.MESG))

        receiving_thread = threading.Thread(target=self.data_handling_loop)  # 데이터 받는 루프
        receiving_thread.start()
        time.sleep(2)

    def data_handling_loop(self):  # 데이터 받아서 저장하는 메서드
        while True:
            data = str(self.sock_lidar.recv(self.BUFF))
            # 라이다에게 데이터 요청 신호를 보냈을 때, 요청을 잘 받았다는 응답을 한 줄 받은 후에 데이터를 받기 시작함
            # 아래 줄은 그 응답 코드을 무시하고 바로 데이터를 받기 위해서 존재함
            if data.__contains__('sEA'): continue

            temp = data.split(' ')[116:477]
            self.data_list = [int(item, 16) for item in temp]

            if self.stop_fg is True: break
        self.sock_lidar.send(str.encode(chr(2) + 'sEN LMDscandata 0' + chr(3)))
        self.sock_lidar.close()

    def stop(self):
        self.stop_fg = True


if __name__ == "__main__":
    current_lidar = Lidar()