# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array

import numpy as np
import cv2
import time

np.set_printoptions(linewidth=10000)

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
