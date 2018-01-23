# 라이다 통신 및 해석(장애물 추출)
# input: LiDAR
# output: numpy array? (to path_planner)

import math
import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Lidar:

    def __init__(self):
        self.HOST = '169.254.248.220'
        self.PORT = 2112
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.fig = plt.figure(figsize = (6, 6))
        self.ax1 = self.fig.add_subplot(1, 1, 1)

    def set_ip(self, ip): self.HOST = ip

    def set_port(self, port): self.PORT = port

    # ROI_tuple: (theta_1, theta_2, radius)
    def set_ROI(self, ROI_tuple): self.ROI = ROI_tuple

    def connect(self):
        self.sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lidar.connect((self.HOST, self.PORT))

    def get_data(self):
        self.sock_lidar.send(str.encode(self.MESG))

        while True:
            data = str(self.sock_lidar.recv(self.BUFF))
            # 라이다에게 데이터 요청 신호를 보냈을 때, 요청을 잘 받았다는 응답을 한 줄 받은 후에 데이터를 받기 시작함
            # 아래 줄은 그 응답 코드을 무시하고 바로 데이터를 받기 위해서 존재함
            if data.__contains__('sEA'): continue

            data_list = data.split(' ')[26:567]

            parsed_data = []

            for i in range(2 * self.ROI[0] + 90, 2 * self.ROI[1] + 91):
                if int(data_list[i], 16) / 10 <= self.ROI[2]:
                    parsed_data.append((int(data_list[i], 16) / 10, 0.5 * i - 45))

            print(parsed_data)

    def animate(self, i, sock_lidar):
        self.sock_lidar.send(str.encode(self.MESG))

        try:
            data = str(sock_lidar.recv(self.BUFF))

            if data.__contains__('sSN'):
                data_array = data.split(' ')[26:567]

                xs = []
                ys = []

                for i in range(2 * self.ROI[0] + 90, 2 * self.ROI[1] + 91):
                    if int(data_array[i], 16) / 10 <= self.ROI[2]:
                        xs.append(int(data_array[i], 16) * math.cos(math.radians(0.5 * i - 45)) / 10)
                        ys.append(int(data_array[i], 16) * math.sin(math.radians(0.5 * i - 45)) / 10)

                # 이전에 찍었던 점들을 모두 지움
                self.ax1.clear()
                # xs와 ys의 index 0, 1에서 노이즈가 발생하기 때문에 index 2부터 plot함
                self.ax1.plot(xs, ys, 'ro', markersize = 2)
                self.ax1.set_xlim(-400, 400)
                self.ax1.set_ylim(-400, 400)
                plt.grid(True)

        except:
            pass

    def plot_data(self):
        try:
            anim = animation.FuncAnimation(self.fig, self.animate, fargs = (self.sock_lidar,), interval = 1)
            plt.show()

        except: pass

current_lidar = Lidar()
current_lidar.set_ROI((0, 180, 400))
current_lidar.connect()
current_lidar.get_data()