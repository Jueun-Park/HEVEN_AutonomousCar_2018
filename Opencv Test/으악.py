import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import (LinearRegression, RANSACRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

######################################°ª ÃÊ±âÈ­######################################
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

# º¯Çü Àü »ç°¢Á¡
L_x1 = 176  # 400
L_x2 = 92
R_x1 = 320  # 560
R_x2 = 447
road_width = R_x2 - L_x2

# º¯Çü ÈÄ »ç°¢Á¡
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



def perspective_Transform():

    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/KATARI2_480_270.mp4')
    except:
        print('카메라 구동 실패')
        return



    while True:
        ret, frame = cap.read()

        if not ret:
            print('비디오 읽기 오류')
            break

        h, w = frame.shape[:2]

        pts1 = np.float32([[L_x1, y1], [R_x1, y1], [L_x2, y2], [R_x2, y2]])
        pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(frame, M, (w,h))

        cv2.imshow('img', img)
        cv2.imshow('frame',frame)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',gray)

        kernel = np.ones((3, 3), np.uint8)

        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Opening',opening)

        gaussian_Blur = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow('Gaussian_Blur',gaussian_Blur)

        Edge1 = cv2.Canny(gray,60,120)
        cv2.imshow('Edge1',Edge1)

        Edge2 = cv2.Canny(opening,60,120)
        cv2.imshow('Edge2',Edge2)

        Edge3 = cv2.Canny(gaussian_Blur,60,120)
        cv2.imshow('Edge3',Edge3)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

perspective_Transform()