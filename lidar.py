# 라이다 통신 및 해석(장애물 추출)
# input: LiDAR
# output: numpy array? (to path_planner)

import math
import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Lidar:

    def __init__(self):
        self.HOST = '192.168.0.95'
        self.PORT = 2111
        self.BUFF = 57600
        self.MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

        self.fig = plt.figure(figsize = (6, 6))
        self.ax1 = self.fig.add_subplot(1, 1, 1)

    def set_ip(self, ip): self.HOST = ip

    def set_port(self, port): self.PORT = port

    def connect(self):
        self.sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_lidar.connect((self.HOST, self.PORT))

    def get_data(self):
        self.sock_lidar.send(str.encode(self.MESG))

        while True:
            data = str(self.sock_lidar.recv(self.BUFF))
            if data.__contains__('sEA'): continue

            data_list = data.split(' ')[26:387]
            print(data_list)

    def animate(self, i, sock_lidar):
        self.sock_lidar.send(str.encode(self.MESG))

        try:
            data = str(sock_lidar.recv(self.BUFF))

            if data.__contains__('sSN'):
                data_array = data.split(' ')[26:387]

                xs = []
                ys = []

                angle = 0
                count = 0

                while angle <= 180:
                    xs.append(int(data_array[count], 16) * math.cos(math.radians(angle)) / 10)
                    ys.append(int(data_array[count], 16) * math.sin(math.radians(angle)) / 10)
                    angle += 0.5
                    count += 1

                self.ax1.clear()
                self.ax1.plot(xs, ys, 'ro', markersize = 2)
                self.ax1.set_xlim([-400, 400])
                self.ax1.set_ylim([-400, 400])

        except:
            pass

    def plot_data(self):
        anim = animation.FuncAnimation(self.fig, self.animate, fargs = (self.sock_lidar,), interval=1)
        plt.show()

current_lidar = Lidar()
current_lidar.connect()
current_lidar.plot_data()