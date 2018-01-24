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

##################################Sub-Functions##########################################

def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = null
    return dst

def camopen(CAM_ID):
    cam = cv2.VideoCapture(CAM_ID)  # 카메라 생성
    if cam.isOpened() == False:  # 카메라 생성 확인
        print('Can\'t open the CAM')
        exit()

    # 카메라 이미지 해상도 얻기
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('size = ', width, height)



    while (True):
        # 카메라에서 이미지 얻기
        ret, frame = cam.read()

        ########### 추가 ########################
        # 이미지를 회전시켜서 img로 돌려받음
        img = Rotate(frame, 270)  # 90 or 180 or 270
        ########################################

        # 얻어온 이미지 윈도우에 표시
        cv2.imshow('CAM_OriginalWindow', frame)

        ########### 추가 ########################
        # 회전된 이미지 표시
        cv2.imshow('CAM_RotateWindow', img)
        #########################################

        # Q 누르기 전까지 작동.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)