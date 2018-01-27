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
'''
안쓰기로 결정 (Gaussian Filter 만 쓰는게 더 좋을듯)
def opening(img):
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('Opening', opening)
    return opening
'''
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

            cv2.line(Edge_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    except:
        pass
    return Edge_img

# ransac
def linear_Ransac(x_points, y_points, y_min, y_max):
    x_points = np.array(x_points)
    y_points = np.array(y_points)

    y_points = y_points.reshape(len(y_points), 1)
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

    try:
        model_ransac.fit(y_points, x_points)
    except ValueError:
        pass
    else:
        line_Y = np.arange(y_min, y_max)
        line_X_ransac = model_ransac.predict(line_Y[:, np.newaxis])

        return line_X_ransac


# ransac
def polynomial_Ransac(x_points, y_points, y_min, y_max):
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    y_points = y_points.reshape(len(y_points), 1)
    model_Sransac = make_pipeline(PolynomialFeatures(2), RANSACRegressor(random_state=42))
    try:
        model_Sransac.fit(y_points, x_points)
    except ValueError:
        pass
    else:
        line_Y = np.arange(y_min, y_max)
        line_X_ransac = model_Sransac.predict(line_Y[:, np.newaxis])
        return line_X_ransac

def extract_Line(dst, img_canny, L_line, R_line):
    global edge_lx, edge_rx
    # draw line roi
    cv2.polylines(dst, np.int32([L_line]), 1, (0, 255, 0), 5)
    cv2.polylines(dst, np.int32([R_line]), 1, (0, 255, 0), 5)

    # canny edge
    L_edge = set_Gray(img_canny, np.int32([L_line]))
    R_edge = set_Gray(img_canny, np.int32([R_line]))

    # separate edge points
    edge_lx, edge_ly = np.where(L_edge >= 255)
    edge_rx, edge_ry = np.where(R_edge >= 255)

    '''# dotted line
    if len(edge_lx) <150 :
        print "left dotted line"
        print len(edge_lx)
    if len(edge_rx) <150 :
        print "right dotted line"
        print len(edge_rx)'''

    for i in range(len(edge_lx)):
        try:
            cv2.circle(dst, (int(edge_ly[i]), int(edge_lx[i])), 1, (0, 155, 255), 2)
        except TypeError:
            pass
    for i in range(len(edge_rx)):
        try:
            cv2.circle(dst, (int(edge_ry[i]), int(edge_rx[i])), 1, (255, 155, 0), 2)
        except TypeError:
            pass
    return dst, edge_lx, edge_ly, edge_rx, edge_ry

# draw straight line
def draw_Straight_Line(dst, L_points, R_points, L_check, R_check, L_num, R_num, L_color, R_color, start_num):
    if L_num == -1:
        draw_Poly(dst, L_check, L_color)
    else:
        draw_Poly(dst, L_points, L_color)
    if R_num == -1:
        draw_Poly(dst, R_check, R_color)
    else:
        draw_Poly(dst, R_points, R_color)
    return dst


# draw poly line
def draw_Poly_Line(dst, L_points, R_points, L_check, R_check, L_num, R_num, L_color, R_color, start_num):
    if L_num == -1:
        draw_Poly(dst, L_check, L_color)
    else:
        draw_Poly(dst, L_points, L_color)
    if R_num == -1:
        draw_Poly(dst, R_check, R_color)
    else:
        draw_Poly(dst, R_points, R_color)
    return dst


# get fit line
def get_Fit_Line(f_lines):
    try:
        if len(f_lines) == 0:
            return None
        elif len(f_lines) == 1:
            lines = lines.reshape(2, 2)
        else:
            lines = np.squeeze(f_lines)
            lines = lines.reshape(lines.shape[0] * 2, 2)
    except:
        return None
    else:
        [vx, vy, x, y] = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
        x1 = 960 - 1  # width of cam(image)
        y1 = int(((960 - x) * vy / vx) + y)
        x2 = 0
        y2 = int((-x * vy / vx) + y)
        result = [x1, y1, x2, y2]
        return result



###############################Main Function#################################





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
    x,y = np.where(Canny >=255)
    print(x)
    print(y)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)




