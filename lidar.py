# 라이다 통신 및 해석(장애물 추출)
# input: LiDAR
# output: numpy array? (to path_planner)

import math
import numpy
import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time

class Lidar:

    def __init__(self):
        self.HOST = '169.254.248.220'
        self.PORT = 2111
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.fig = plt.figure(figsize = (6, 6))
        self.ax1 = self.fig.add_subplot(1, 1, 1)

        self.data_list = []

    def set_ip(self, ip): self.HOST = ip

    def set_port(self, port): self.PORT = port

    # ROI_tuple: (theta_1, theta_2, radius, width, length)
    def set_ROI(self, ROI_tuple):
        self.ROI = ROI_tuple
        self.xmin = self.ROI[2] * math.cos(math.radians(self.ROI[1]))
        self.xmax = self.ROI[2] * math.cos(math.radians(self.ROI[0]))

        self.x1 = numpy.arange(self.xmin, 0.05, 0.05)
        self.x2 = numpy.arange(0, self.xmax + 0.05, 0.05)
        self.x3 = numpy.arange(self.xmin, self.xmax + 0.05, 0.05)

        self.y1 = [(math.tan(math.radians(self.ROI[1])) * v) for v in self.x1]
        self.y2 = [(math.tan(math.radians(self.ROI[0])) * v) for v in self.x2]
        self.y3 = [(math.sqrt(self.ROI[2] ** 2 - v ** 2)) for v in self.x3]

        #Linear Equation x-axis
        self.x4 = numpy.arange(-self.ROI[3] / 2, self.ROI[3] / 2 + 0.05, 0.05)

        self.y4 = numpy.array([0 for i in self.x4])
        self.y5 = numpy.array([self.ROI[4] for i in self.x4])
        self.y6 = numpy.array([self.ROI[4] * 2 / 3 for i in self.x4])
        self.y7 = numpy.array([self.ROI[4] / 3 for i in self.x4])

        #Linear Equation y-axis
        self.y8 = numpy.arange(0, self.ROI[4] + 0.05, 0.05)

        self.x5 = numpy.array([-self.ROI[3] / 2 for i in self.y8])
        self.x6 = numpy.array([self.ROI[3] / 2 for i in self.y8])

    def loop(self):
        self.sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lidar.connect((self.HOST, self.PORT))
        self.sock_lidar.send(str.encode(self.MESG))

        while True:
            data = str(self.sock_lidar.recv(self.BUFF))
            # 라이다에게 데이터 요청 신호를 보냈을 때, 요청을 잘 받았다는 응답을 한 줄 받은 후에 데이터를 받기 시작함
            # 아래 줄은 그 응답 코드을 무시하고 바로 데이터를 받기 위해서 존재함
            if data.__contains__('sEA'): continue

            self.data_list = data.split(' ')[26:567]

    def initiate(self):
        t = threading.Thread(target = self.loop)
        t.start()

    def get_data(self):
        danger = []
        look_out = []
        object_dectected = []

        for n in range(2 * self.ROI[0] + 90, 2 * self.ROI[1] + 91):
            x = int(self.data_list[n], 16) * math.cos(math.radians(0.5 * n - 45)) / 10
            y = int(self.data_list[n], 16) * math.sin(math.radians(0.5 * n - 45)) / 10
            if 3 <= int(self.data_list[n], 16) / 10 and int(self.data_list[n], 16) / 10 <= self.ROI[2]\
                    and abs(x) <= self.ROI[3] / 2 and y <= self.ROI[4]:
                if y <= self.ROI[4]/3:
                    danger.append((x,y))
                elif y <= self.ROI[4] * 2 / 3:
                    look_out.append((x,y))
                elif y <= self.ROI[4] / 3:
                    object_dectected.append((x,y))
                else:
                    continue

        print(danger)
        print(look_out)
        print(object_dectected)
        #parsed_data.append((int(self.data_list[n], 16) / 10, -45 + 0.5 * n))
        #return parsed_data

    def animate(self, i):
        try:
            xs = []
            ys = []

            for n in range(2 * self.ROI[0] + 90, 2 * self.ROI[1] + 91):
                if 3 <= int(self.data_list[n], 16) / 10 and int(self.data_list[n], 16) / 10 <= self.ROI[2]:
                    xs.append(int(self.data_list[n], 16) * math.cos(math.radians(0.5 * n - 45)) / 10)
                    ys.append(int(self.data_list[n], 16) * math.sin(math.radians(0.5 * n - 45)) / 10)

            # 이전에 찍었던 점들을 모두 지움
            self.ax1.clear()

            self.ax1.plot(xs, ys, 'ro', markersize = 2)

            # ROI 경계선 그리기: 개발자가 보기 편하도록
            self.ax1.plot(self.x1, self.y1, 'b', linewidth = 1)
            self.ax1.plot(self.x2, self.y2, 'b', linewidth = 1)
            self.ax1.plot(self.x3, self.y3, 'b', linewidth = 1)

            self.ax1.plot(self.x4, self.y4, 'r', linewidth = 1)
            self.ax1.plot(self.x4, self.y5, 'r', linewidth = 1)
            self.ax1.plot(self.x4, self.y6, 'r', linewidth = 1)
            self.ax1.plot(self.x4, self.y7, 'r', linewidth = 1)

            self.ax1.plot(self.x5, self.y8, 'r', linewidth = 1)
            self.ax1.plot(self.x6, self.y8, 'r', linewidth = 1)

            # 축 범위 지정하기
            self.ax1.set_xlim(-100, 100)
            self.ax1.set_ylim(0, 200)

        except:
            pass

    def plot_data(self):
        anim = animation.FuncAnimation(self.fig, self.animate, interval = 1)
        plt.show()


if __name__ == "__main__":
    current_lidar = Lidar()
    current_lidar.set_ROI((0, 180, 80, 80, 300))
    current_lidar.initiate()
    current_lidar.plot_data()