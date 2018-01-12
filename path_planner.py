# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 경로가 표시된 numpy array (to car_control)

import numpy

class parabola:

    def __init__(self, a0, a1, a2):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    def plot(self, color):
        x = np.arange(-200, 0)
        y = [self.a0 + self.a1 * value + self.a2 * value ** 2 for value in x]
        plt.plot(x, y, color)

    def get_curvature(self, x):
        curvature = 2 * self.a2 / pow((1 + (self.a1 + 2 * self.a2 * x) ** 2), 1.5)
        return curvature

    def get_value(self, x):
        value = self.a0 + self.a1 * x + self.a2 * x**2
        return value