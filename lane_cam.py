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
'''
# set cross point (Rotation 때문에 저번대회랑 다름)
y1 = 185
y2 = 269

# 원래 Pixel (Rotation 때문에 저번대회랑 다름)
L_x1 = 176  # 400
L_x2 = 92
R_x1 = 320  # 560
R_x2 = 447
road_width = R_x2 - L_x2

# 바꿀 Pixel (Rotation 때문에 저번대회랑 다름)
Ax1 = 85 + 5  # 50
Ax2 = 215 - 5  # 470
Ay1 = 0
Ay2 = 570

'''
# set cross point (Rotation 때문에 저번대회랑 다름)
x1 = 185
x2 = 269

# 원래 Pixel (Rotation 때문에 저번대회랑 다름)
L_y1 = 304  # 400
L_y2 = 388
R_y1 = 160  # 560
R_y2 = 3
road_width = R_y2 - L_y2

# 바꿀 Pixel (Rotation 때문에 저번대회랑 다름)
Ax1 = 85 + 5  # 50
Ax2 = 215 - 5  # 470
Ay1 = 0
Ay2 = 570

# Homograpy transform
#pts1 = np.float32([[x1, L_y1], [x1, R_y1], [x2, L_y2], [x2, R_y2]])
#pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])
pts1 = np.float32([[185, 304], [485, 160], [269, 3], [269, 388]])
pts2 = np.float32([[90, 570], [90, 0], [210, 0], [210, 570]])
M = cv2.getPerspectiveTransform(pts1, pts2)
i_M = cv2.getPerspectiveTransform(pts2, pts1)

real_Road_Width = 125
#########################################################################################
##################################Sub-Functions##########################################


#Filter Functions
def set_Gray(img, region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, (255, 255, 255))
    img_ROI = cv2.bitwise_and(img, mask)
    #cv2.imshow('img_ROI',img_ROI)
    return img_ROI

def set_Red(img, region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, (0, 0, 255))
    img_red = cv2.bitwise_and(img, mask)
    #cv2.imshow('img_red',img_red)
    return img_red

def BGR2HSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    hsv = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow('hsv_Cvt',hsv)
    return hsv

def gaussian_Blur(img):
    blur = cv2.GaussianBlur(img, (3,3), 0)
    #cv2.imshow('Blur',blur)
    return blur

def opening(img):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('Opening', opening)
    return opening

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



CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/0507_one_lap_normal.mp4'
#CAM_ID = 0
cam = cv2.VideoCapture(CAM_ID)  # 카메라 생성
if cam.isOpened() == False:  # 카메라 생성 확인
    print('Can\'t open the CAM')


# 카메라 이미지 해상도 얻기
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('size = ', width, height)



while (True):
    # 카메라에서 이미지 얻기
    ret, frame = cam.read()

    # 이미지를 회전시켜서 rotated로 돌려받음
    rotated = Rotate(frame, 270)  # 90 or 180 or 270
    ########################################
    cv2.imshow('ORIGINAL',frame)
    cv2.imshow('ROTATED',rotated)

    blur_img = gaussian_Blur(rotated)
    open_img = opening(rotated)
    blur_hsv = BGR2HSV(blur_img)
    open_hsv = BGR2HSV(open_img)
    nothing = BGR2HSV(rotated)
    blur_hsv = cv2.Canny(blur_hsv, 70, 140)
    open_hsv = cv2.Canny(open_hsv, 70, 140)
    nothing = cv2.Canny(nothing, 70, 140)


    cv2.imshow('blur',blur_img)
    cv2.imshow('opening', open_img)
    cv2.imshow('blur_hsv',blur_hsv)
    cv2.imshow('open_hsv',open_hsv)
    cv2.imshow('nothing', nothing)

    height, width = rotated.shape[:2]

    dst = cv2.warpPerspective(rotated, M, (width, height))
    cv2.imshow('dst',dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)




