# -*- Encoding:UTF-8 -*- #

# 카메라 통신 및 차선 인식
# input: sign_cam
# output: numpy array? (to path_planner)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import (LinearRegression, RANSACRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


######################################변수 선언#########################################
global direction, L_num, R_num, L_ransac, R_ransac, L_roi, R_roi, start_num, L_error, R_error
global frame_num, L_check, R_check, stop_Lines, destination_J, destination_I
global L_E, R_E, mid_ransac, lane_width
global edge_lx, edge_rx  # dotted line detection

destination_J = 39
stop_Lines = 0
frame_num = 0
bird_height = 480
bird_width = 270
height = 270
width = 480
height_ROI = 270
L_num = 0
R_num = 0
L_ransac = 0
R_ransac = 0
L_check = 0
R_check = 0
L_roi = 0
R_roi = 0
L_error = 0
R_error = 0
num_y = bird_height - height_ROI
direction = 'straight'
num = 0
start_num = 0
L_E = 0
R_E = 0
mid_ransac = 135.
lane_width = 30

# set cross point
y1 = 185
y2 = 269

# 원래 Pixel
L_x1 = 176  # 400
L_x2 = 92
R_x1 = 320  # 560
R_x2 = 447
road_width = R_x2 - L_x2

# 바꿀 Pixel
Ax1 = 85 + 5  # 50
Ax2 = 215 - 5  # 470
Ay1 = 0
Ay2 = 570

# Homograpy transform
pts1 = np.float32([[L_x1, y1], [R_x1, y1], [L_x2, y2], [R_x2, y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])
M = cv2.getPerspectiveTransform(pts1, pts2)
i_M = cv2.getPerspectiveTransform(pts2, pts1)

real_Road_Width = 125