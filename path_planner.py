# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 경로가 표시된 numpy array (to car_control)

import numpy as np
import matplotlib.pyplot as plt


class Parabola:

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
        value = self.a0 + self.a1 * x + self.a2 * x ** 2
        return value


class Lane:

    def __init__(self, left_coeffs, right_coeffs):
        self.left_coeffs = left_coeffs
        self.right_coeffs = right_coeffs
        self.trajectory = Parabola((left_coeffs[0] + right_coeffs[0]) / 2,
                                   (left_coeffs[1] + right_coeffs[1]) / 2, (left_coeffs[2] + right_coeffs[2]) / 2)

    def set_trajectory(self, ratio_1, ratio_2):
        self.trajectory = Parabola(
            (self.left_coeffs[0] * ratio_1 + self.right_coeffs[0] * ratio_2) / (ratio_1 + ratio_2),
            (self.left_coeffs[1] * ratio_1 + self.right_coeffs[1] * ratio_2) / (ratio_1 + ratio_2),
            (self.left_coeffs[2] * ratio_1 + self.right_coeffs[2] * ratio_2) / (ratio_1 + ratio_2))

    def get_trajectory(self): return self.trajectory
