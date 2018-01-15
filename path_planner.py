# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array

import numpy as np
import matplotlib.pyplot as plt


class Parabola:

    def __init__(self, a0, a1, a2):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    def get_curvature(self, x):
        curvature = 2 * self.a2 / pow((1 + (self.a1 + 2 * self.a2 * x) ** 2), 1.5)
        return curvature

    def get_derivative(self, x):
        derivative = self.a1 = 2 * self.a2 * x
        return derivative

    def get_value(self, x):
        value = self.a0 + self.a1 * x + self.a2 * x ** 2
        return value


class Lane:

    def __init__(self, left_coeffs, right_coeffs):
        self.left_coeffs = left_coeffs
        self.right_coeffs = right_coeffs

        self.trajectory = Parabola(
            (left_coeffs[0] + right_coeffs[0]) / 2,
            (left_coeffs[1] + right_coeffs[1]) / 2,
            (left_coeffs[2] + right_coeffs[2]) / 2)

    def set_coeffs(self, left_coeffs, right_coeffs):
        self.left_coeffs = left_coeffs
        self.right_coeffs = right_coeffs

    def set_trajectory(self, ratio_1, ratio_2):
        self.trajectory = Parabola(
            (self.left_coeffs[0] * ratio_1 + self.right_coeffs[0] * ratio_2) / (ratio_1 + ratio_2),
            (self.left_coeffs[1] * ratio_1 + self.right_coeffs[1] * ratio_2) / (ratio_1 + ratio_2),
            (self.left_coeffs[2] * ratio_1 + self.right_coeffs[2] * ratio_2) / (ratio_1 + ratio_2))

    def get_trajectory(self): return self.trajectory

    def plot_lanes(self):
        x = np.arange(-200, 0)
        left = [self.left_coeffs[0] + self.left_coeffs[1] * value + self.left_coeffs[2] * value ** 2 for value in x]
        right = [self.right_coeffs[0] + self.right_coeffs[1] * value + self.right_coeffs[2] * value ** 2 for value in x]
        center = [self.trajectory.a0 + self.trajectory.a1 * value + self.trajectory.a2 * value ** 2 for value in x]

        plt.plot(x, left)
        plt.plot(x, right)
        plt.plot(x, center)

    def get_lane_status(self):
        measure_location = -25
        lane_center = self.get_trajectory().get_value(measure_location)
        lane_slope = self.get_trajectory().get_derivative(measure_location)
        lane_curvature = self.get_trajectory().get_curvature(measure_location)

        lane_status = np.array([lane_center, lane_slope, lane_curvature])
        return lane_status


def plot_curve(lanes):  # lanes: Lane instance
    plt.figure(figsize=(6, 6))
    plt.ylim(-100, 100)
    plt.xlim(-200, 0)

    lanes.plot_lanes()

    plt.show()

left_coeffs = np.array([60, -0.05, 0.0005])
right_coeffs = np.array([-60, -0.025, 0.00075])

current_lane = Lane(left_coeffs, right_coeffs)
plot_curve(current_lane)
print(current_lane.get_lane_status())

