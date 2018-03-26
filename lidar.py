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
    RADIUS = 300  # 원일 경우 지름, 사각형일 경우 한 변

    def __init__(self):
        self.HOST = '169.254.248.220'
        self.PORT = 2111
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.mode = modes['STATIC_OBS']

        self.data_list = []  # 데이터가 16진수 통으로 들어갈 것이다

    def loop(self):  # 데이터 받아서 저장하는 메서드
        self.sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lidar.connect((self.HOST, self.PORT))
        self.sock_lidar.send(str.encode(self.MESG))

        while True:
            data = str(self.sock_lidar.recv(self.BUFF))
            # 라이다에게 데이터 요청 신호를 보냈을 때, 요청을 잘 받았다는 응답을 한 줄 받은 후에 데이터를 받기 시작함
            # 아래 줄은 그 응답 코드을 무시하고 바로 데이터를 받기 위해서 존재함
            if data.__contains__('sEA'): continue

            self.data_list = data.split(' ')[116:477]

    def animation_loop(self):  # 저장한 데이터 그림 그려주는 메서드
        while True:
            canvas = np.full((self.RADIUS * 2, self.RADIUS, 3), 255, np.uint8)  # 내가 곧 그림을 그릴 곳 (넘파이어레이)

            points = np.full((361, 2), -1000, np.int)  # 점 찍을 좌표들을 담을 어레이 (x, y), 멀리 -1000 으로 채워둠.

            for angle in range(0, 361):
                r = int(self.data_list[angle], 16) / 10  # 차에서 장애물까지의 거리, 단위는 cm

                if 1.0 <= r <= self.RADIUS:  # 라이다 바로 앞 1cm 의 노이즈는 무시

                    # r-theta 를 x-y 로 바꿔서 (실제에서의 위치, 단위는 cm)
                    x = r * math.cos(math.radians(0.5 * angle))
                    y = r * math.sin(math.radians(0.5 * angle))

                    # 좌표 변환, 화면에서 보이는 좌표(왼쪽 위가 (0, 0))에 맞춰서 집어넣는다
                    points[angle][0] = round(x) + self.RADIUS
                    points[angle][1] = self.RADIUS - round(y)

            for point in points:  # 장애물들에 대하여
                cv2.circle(canvas, tuple(point), 2, (0, 0, 255), -1)  # 캔버스에 점 찍기

            cv2.imshow('lidar', canvas)  # 창 띄워서 확인

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    def initiate(self):  # 루프 시작
        receiving_thread = threading.Thread(target=self.loop)  # 데이터 받는 루프
        animation_thread = threading.Thread(target=self.animation_loop)  # 창 띄워서 장애물 보여주기 루프

        receiving_thread.start()
        time.sleep(2)
        animation_thread.start()


if __name__ == "__main__":
    current_lidar = Lidar()
    current_lidar.initiate()
