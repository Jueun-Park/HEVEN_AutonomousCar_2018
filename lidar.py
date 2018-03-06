# 라이다 통신 및 해석(장애물 추출)
# input: LiDAR
# output: numpy array? (to path_planner)

import math
import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import threading
import time

modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
         'MOVING_OBS': 3,'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

class Lidar:

    # 미션별 ROI에 쓰이는 상수와 Plotting에 쓰이는 선분들
    # 1) 정적 장애물
    STATIC_WIDTH = 60
    STATIC_HEIGHT = 50

    # 2) 동적 장애물
    MOVING_WIDTH = 70
    MOVING_HEIGHT = 30

    # 3) S_curve
    S_WIDTH = 80
    S_HEIGHT = 20

    # 4) Narrow
    NARROW_WIDTH = 70
    NARROW_HEIGHT = 70

    line0 = Line2D([-50, -50, 50, 50], [-100, -4, -4, -100], color = 'k')

    line1 = Line2D([-STATIC_WIDTH, -STATIC_WIDTH, STATIC_WIDTH, STATIC_WIDTH, -STATIC_WIDTH],
                   [0, STATIC_HEIGHT * 3, STATIC_HEIGHT * 3, 0, 0], color = 'b')
    line2 = Line2D([-STATIC_WIDTH, STATIC_WIDTH], [STATIC_HEIGHT, STATIC_HEIGHT], color = 'b')
    line3 = Line2D([-STATIC_WIDTH, STATIC_WIDTH], [STATIC_HEIGHT * 2, STATIC_HEIGHT * 2], color = 'b')

    line4 = Line2D([-MOVING_WIDTH, -MOVING_WIDTH, MOVING_WIDTH, MOVING_WIDTH, -MOVING_WIDTH],
                   [0, MOVING_HEIGHT, MOVING_HEIGHT, 0, 0], color = 'b')
    line5 = Line2D([-S_WIDTH, -S_WIDTH, S_WIDTH, S_WIDTH, -S_WIDTH], [0, S_HEIGHT, S_HEIGHT, 0, 0], color = 'b')

    line6 = Line2D([-NARROW_WIDTH, -NARROW_WIDTH, NARROW_WIDTH, NARROW_WIDTH, -NARROW_WIDTH],
                   [0, NARROW_HEIGHT * 3, NARROW_HEIGHT * 3, 0, 0], color = 'b')
    line7 = Line2D([-NARROW_WIDTH, NARROW_WIDTH], [NARROW_HEIGHT, NARROW_HEIGHT], color = 'b')
    line8 = Line2D([-NARROW_WIDTH, NARROW_WIDTH], [NARROW_HEIGHT * 2, NARROW_HEIGHT * 2], color = 'b')

    def __init__(self):
        self.HOST = '169.254.248.220'
        self.PORT = 2111
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.mode = modes['STATIC_OBS']

        self.fig = plt.figure(figsize = (6, 6))
        self.ax1 = self.fig.add_subplot(1, 1, 1)

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

            if self.mode == modes['STATIC_OBS']:
                try:
                    self.parsed_data = [[], [], []]

                    for i in range(0, 361):
                        r = int(self.data_list[i], 16) / 10

                        if r >= 1:
                            x = r * math.cos(math.radians(0.5 * i))
                            y = r * math.sin(math.radians(0.5 * i))

                            if -self.STATIC_WIDTH <= x <= self.STATIC_WIDTH and y < self.STATIC_HEIGHT * 3:
                                self.parsed_data[int(y / self.STATIC_HEIGHT)].append((x, y))
                except: pass

            elif self.mode == modes['MOVING_OBS']:
                try:
                    self.parsed_data = [[]]

                    for i in range(0, 361):
                        r = int(self.data_list[i], 16) / 10

                        if r >= 1:
                            x = r * math.cos(math.radians(0.5 * i))
                            y = r * math.sin(math.radians(0.5 * i))

                            if -self.MOVING_WIDTH <= x <= self.MOVING_WIDTH and y < self.MOVING_HEIGHT:
                                self.parsed_data[0].append((x, y))
                except: pass

            elif self.mode == modes['S_CURVE']:
                try:
                    self.parsed_data = [[]]

                    for i in range(0, 361):
                        r = int(self.data_list[i], 16) / 10

                        if r >= 1:
                            x = r * math.cos(math.radians(0.5 * i))
                            y = r * math.sin(math.radians(0.5 * i))

                            if -self.S_WIDTH <= x <= self.S_WIDTH and y < self.S_HEIGHT:
                                self.parsed_data[0].append((x, y))
                except: pass

            elif self.mode == modes['NARROW']:
                try:
                    self.parsed_data = [[], [], []]

                    for i in range(0, 361):
                        r = int(self.data_list[i], 16) / 10

                        if r >= 1:
                            x = r * math.cos(math.radians(0.5 * i))
                            y = r * math.sin(math.radians(0.5 * i))

                            if -self.NARROW_WIDTH <= x <= self.NARROW_WIDTH and y < self.NARROW_HEIGHT * 3:
                                self.parsed_data[int(y / self.NARROW_HEIGHT)].append((x, y))
                except: pass

    def initiate(self):
        receiving_thread = threading.Thread(target = self.loop)
        receiving_thread.start()

    def get_data(self): return self.parsed_data

    def animate(self, i):
        try:
            xs = []
            ys = []

            for n in range(0, 361):
                r = int(self.data_list[n], 16) / 10

                if r >= 1:
                    x = r * math.cos(math.radians(0.5 * n))
                    y = r * math.sin(math.radians(0.5 * n))

                    xs.append(x)
                    ys.append(y)

            self.ax1.clear()
            self.ax1.plot(xs, ys, 'ro', markersize = 2, zorder = 10)

            # ROI 경계선 그리기: 개발자가 보기 편하도록
            self.ax1.add_line(self.line0)

            if self.mode == modes['STATIC_OBS']:
                self.ax1.add_line(self.line1)
                self.ax1.add_line(self.line2)
                self.ax1.add_line(self.line3)

            elif self.mode == modes['MOVING_OBS']:
                self.ax1.add_line(self.line4)

            elif self.mode == modes['S_CURVE']:
                self.ax1.add_line(self.line5)

            elif self.mode == modes['NARROW']:
                self.ax1.add_line(self.line6)
                self.ax1.add_line(self.line7)
                self.ax1.add_line(self.line8)

            # 축 범위 지정하기
            self.ax1.set_xlim(-200, 200)
            self.ax1.set_ylim(-100, 300)

        except: pass

    def plot_data(self):
        anim = animation.FuncAnimation(self.fig, self.animate, interval = 1)
        plt.show()


if __name__ == "__main__":
    current_lidar = Lidar()
    current_lidar.initiate()
    current_lidar.set_mode(modes['NARROW'])

    while True:
        print(current_lidar.get_data())

    #current_lidar.plot_data()