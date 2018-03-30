# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array

import numpy as np
import cv2
import threading
from lidar import Lidar
import time

np.set_printoptions(linewidth=1000)

class MotionPlanner():

    def __init__(self, lidar_instance):
        self.lidar = lidar_instance


    def loop(self):
        Rad=current_lidar.RADIUS
        while True:
            t1 = time.time()
            data = np.zeros((2, 37), np.int)
            current_frame = self.lidar.frame

            if current_frame is not None:


                for r in range(0, Rad):
                    for theta in range(0, 181, 5):
                        x = Rad + int(round(r * np.cos(np.radians(theta)))) - 1
                        y = Rad - int(round(r * np.sin(np.radians(theta)))) - 1

                        if data[0][int(theta / 5)] == 0:
                            data[1][int(theta / 5)] = r

                        if current_frame[y][x] != 0:
                            data[0][int(theta / 5)] = 1
                for i in range(0, 37):
                    x = Rad + int(round(data[1][i] * np.cos(np.radians(i * 5)))) - 1
                    y = Rad - int(round(data[1][i] * np.sin(np.radians(i * 5)))) - 1
                    cv2.line(current_frame, (Rad, Rad), (x, y), 255)

                cv2.imshow('lidar', current_frame)
                t2 = time.time()
                print(t2 - t1)

            if cv2.waitKey(1) & 0xFF == ord('q'): break


    def initiate(self):
        thread = threading.Thread(target=self.loop)
        thread.start()



current_lidar = Lidar()
current_lidar.initiate()

motion_plan = MotionPlanner(current_lidar)
motion_plan.initiate()

'''
canvas = np.zeros((400, 400), np.uint8)
data = np.zeros((2, 19), np.int)

cv2.circle(canvas, (300, 150), 70, 255, -1)

for r in range(0, 600):
    for theta in range(0, 91, 5):
        x = int(r * np.cos(np.radians(theta)))
        y = int(r * np.sin(np.radians(theta)))

        if x >= 400 or y >= 400: continue

        if data[0][int(theta / 5)] == 0:
            data[1][int(theta / 5)] = r

        if canvas[y][x] != 0:
            data[0][int(theta / 5)] = 1

print(data)

for i in range(0, 19):
    x = int(data[1][i] * np.cos(np.radians(i * 5)))
    y = int(data[1][i] * np.sin(np.radians(i * 5)))
    cv2.line(canvas, (0, 0), (x, y), 255)

while True:
    cv2.imshow('canvas', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'): break
'''