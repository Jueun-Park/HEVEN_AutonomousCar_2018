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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
from matplotlib import style


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
저번대회의 코드
(Rotation 을 적용하지 않은 경우)
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
L_y1 = 320
L_y2 = 479
R_y1 = 160
R_y2 = 1
road_width = R_y2 - L_y2

# 바꿀 Pixel (Rotation 때문에 저번대회랑 다름)
Ax1 = 0
Ax2 = 480
Ay1 = 210
Ay2 = 60

# Homograpy transform
pts1 = np.float32([[x1, L_y1], [x1, R_y1], [x2, R_y2], [x2, L_y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax1, Ay2], [Ax2, Ay2], [Ax2, Ay1]])
#pts1 = np.float32([[185, 320], [185, 160], [269, 1], [269, 479]])
#pts2 = np.float32([[0, 210], [0, 60], [480, 60], [480, 210]])
M = cv2.getPerspectiveTransform(pts1, pts2)
i_M = cv2.getPerspectiveTransform(pts2, pts1)

real_Road_Width = 125

#########################################################################################
##################################Sub-Functions##########################################


#Filter Function

# BGR image 를 HSV image 로 변환하여 차선만 잘 보이게 함.
def BGR2HSV(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160]) # 이 Lower 값을 조절하여 날씨에 대한 대응 가능.
    upper = np.array([255, 255, 255])
    #Lower, Upper 에서 건드리는 건 hsv 중 v(Value)값임.[명도]
    mask = cv2.inRange(img_hsv, lower, upper)
    hsv = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow('hsv_Cvt',hsv)
    return hsv

#Gaussian Blur Filter 를 씌워 Noise 를 없앰.
def gaussian_Blur(img):
    blur = cv2.GaussianBlur(img, (3,3), 0)  #여기 (3,3)은 kernel 값. 조절 가능(Only 홀수)
    #cv2.imshow('Blur',blur)
    return blur
'''
안쓰기로 결정 (Gaussian Filter 만 쓰는게 더 좋을듯)
def opening(img):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('Opening', opening)
    return opening
'''

# 원하는 각도로 영상을 Rotate 시킬 수 있음.
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

def houghLines(Edge_img):

    lines = cv2.HoughLines(Edge_img, 1, np.pi / 180, 200)

    try:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(Edge_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except:
        pass
    return Edge_img

def houghLinesP(Edge_img):

    minLineLength = 100
    maxLineGap = 10

    try:
        lines = cv2.HoughLinesP(Edge_img, 1, np.pi / 360, 100, minLineLength, maxLineGap)
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(Edge_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    except:
        pass
    return Edge_img

def lane_Roi(dst, direction, L_num, R_num, L_ransac, R_ransac, L_roi_before, R_roi_before):
    # left line roi
    try:
        if L_num != 0:
            if direction == 'left' or direction == 'right':
                L_roi = np.array([[(int(L_ransac[0]) - 25, height_ROI), (int(L_ransac[0]) + 25, height_ROI),
                                   (int(L_ransac[num_y // 3]) + 25, height_ROI + num_y // 3), (80, bird_height - 60),
                                   (20, bird_height - 60), (int(L_ransac[num_y // 3]) - 25, height_ROI + num_y // 3)]])
            else:
                L_roi = np.array([[(int(L_ransac[0]) - 25, height_ROI), (int(L_ransac[0]) + 25, height_ROI),
                                   (int(L_ransac[num_y // 3]) + 25, height_ROI + num_y // 3),
                                   (int(L_ransac[num_y - 50]) + 25, bird_height - 60),
                                   (int(L_ransac[num_y - 50]) - 25, bird_height - 60),
                                   (int(L_ransac[num_y // 3]) - 25, height_ROI + num_y // 3)]])
        elif direction == 'straight':
            L_roi = np.array([[(0, 280), (bird_width / 2 - 40, 280), (bird_width / 2 - 40, height_ROI + num_y / 2),
                               (bird_width / 2 - 40, bird_height - 65), (15, bird_height - 65)]])
        elif direction == 'right':
            L_roi = L_roi_before
        elif direction == 'left':
            L_roi = L_roi_before
        else:
            L_roi = L_roi_before

    except TypeError:
        L_roi = np.array([[(0, 280), (bird_width / 2 - 40, 280), (bird_width / 2 - 40, height_ROI + num_y / 2),
                           (bird_width / 2 - 40, bird_height - 65), (15, bird_height - 65)]])

    # right line roi
    try:
        if R_num != 0:
            if direction == 'left' or direction == 'right':
                R_roi = np.array([[(250, bird_height - 60), (190, bird_height - 60),
                                   (int(R_ransac[num_y // 3]) - 25, height_ROI + num_y // 3),
                                   (int(R_ransac[0]) - 25, height_ROI),
                                   (int(R_ransac[0]) + 25, height_ROI),
                                   (int(R_ransac[num_y // 3]) + 25, height_ROI + num_y // 3)]])
            else:
                R_roi = np.array([[(int(R_ransac[num_y - 100]) + 25, bird_height - 60),
                                   (int(R_ransac[num_y - 50]) - 25, bird_height - 60),
                                   (int(R_ransac[num_y // 3]) - 25, height_ROI + num_y // 3),
                                   (int(R_ransac[0]) - 25, height_ROI),
                                   (int(R_ransac[0]) + 25, height_ROI),
                                   (int(R_ransac[num_y // 3]) + 25, height_ROI + num_y // 3)]])

        elif direction == 'straight':
            R_roi = np.array([[(bird_width - 15, bird_height - 65), (bird_width / 2 + 40, bird_height - 65),
                               (bird_width / 2 + 40, height_ROI + num_y / 2), (bird_width / 2 + 40, 280),
                               (bird_width, 280)]])

        elif direction == 'right':
            R_roi = R_roi_before

        elif direction == 'left':
            R_roi = R_roi_before
        else:
            R_roi = R_roi_before
    except TypeError:
        R_roi = np.array([[(bird_width - 15, bird_height - 65), (bird_width / 2 + 40, bird_height - 65),
                           (bird_width / 2 + 40, height_ROI + num_y / 2), (bird_width / 2 + 40, 280),
                           (bird_width, 280)]])
    return L_roi, R_roi

def lane_Extract(dst, img_canny, L_line, R_line):

    return dst, edge_lx, edge_ly, edge_rx, edge_ry


def poly_Ransac(x_points, y_points, y_min, y_max):


    return

###############################Main Function#################################

def lane_Detection(img):
    #일단, 보여줄 건 추출한 lane을 보여주고, 그에 따른 이차함수의 계수 3개를 보여줌.

    return coef1, coef2, coef3
    #이차함수 꼴 = coef1*(x^2) + coef2*x + coef3




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
    #cv2.imshow('ORIGINAL',frame)
    #cv2.imshow('ROTATED',rotated)
    height, width = rotated.shape[:2]
    dst = cv2.warpPerspective(rotated, M, (height, width))
    #cv2.imshow('dst',dst)
    blur_img = gaussian_Blur(dst)
    #cv2.imshow('blur',blur_img)
    hsv = BGR2HSV(blur_img)
    #cv2.imshow('blur_hsv',hsv)
    Canny = cv2.Canny(hsv, 40, 80)
    #cv2.imshow('hsv_Canny', Canny)
    Houghed = houghLines(Canny)
    #cv2.imshow('hough', Houghed)
    HoughedP = houghLinesP(Canny)
    #cv2.imshow('houghP',HoughedP)

    X,y = np.where(Canny >=255)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)




