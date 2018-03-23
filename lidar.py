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
         'MOVING_OBS': 3,'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

class Lidar:
    
    def __init__(self):
        self.HOST = '169.254.248.220'
        self.PORT = 2111
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.mode = modes['STATIC_OBS']

        self.data_list = []
        self.parsed_data = []

    def set_mode(self, mode): self.mode = mode

    def loop(self):
        self.sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lidar.connect((self.HOST, self.PORT))
        self.sock_lidar.send(str.encode(self.MESG))

        while True:
            data = str(self.sock_lidar.recv(self.BUFF))
            # 라이다에게 데이터 요청 신호를 보냈을 때, 요청을 잘 받았다는 응답을 한 줄 받은 후에 데이터를 받기 시작함
            # 아래 줄은 그 응답 코드을 무시하고 바로 데이터를 받기 위해서 존재함
            if data.__contains__('sEA'): continue

            self.data_list = data.split(' ')[116:477]


    def animation_loop(self):
        while True:
            canvas = np.full((600, 600, 3), 255, np.uint8)

            points = np.zeros((361, 2), np.int)

            for i in range(0, 361):
                r = int(self.data_list[i], 16) / 10

                if r >= 1:
                    x = -r * math.cos(math.radians(0.5 * i))
                    y = r * math.sin(math.radians(0.5 * i))

                    points[i][0] = round(x) + 300
                    points[i][1] = 600 - round(y)

                else:
                    points[i][0] = -100000
                    points[i][1] = -100000

            for point in points:
                cv2.circle(canvas, tuple(point), 60, (0, 0, 255), -1)

            cv2.imshow('lidar', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    def initiate(self):
        receiving_thread = threading.Thread(target = self.loop)
        animation_thread = threading.Thread(target=self.animation_loop)

        receiving_thread.start()
        time.sleep(1)
        animation_thread.start()

    def get_data(self): return self.parsed_data

if __name__ == "__main__":
    current_lidar = Lidar()
    current_lidar.initiate()
    current_lidar.set_mode(modes['NARROW'])
