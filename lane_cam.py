# -*- Encoding:UTF-8 -*- #

# 카메라 통신 및 차선 인식
# input: sign_cam
# output: 이차함수의 계수.

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
from numpy.polynomial.polynomial import polyfit

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


#############################480x270###################################
#(Rotation 을 적용하지 않은 경우)
# set cross point
y1 = 185
y2 = 269

# 원래 Pixel
L_x1 = 160  # 400
L_x2 = 0
R_x1 = 320  # 560
R_x2 = 480
road_width = R_x2 - L_x2

# 바꿀 Pixel
Ax1 = 60  # 50
Ax2 = 210  # 470
Ay1 = 0
Ay2 = 480

pts1 = np.float32([[L_x1, y1], [R_x1, y1], [L_x2, y2], [R_x2, y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])
#########################################################################
'''
#################################432x240#################################
# set cross point 
y1 = 160
y2 = 239

# 원래 Pixel 
L_x1 = 100  # 400
L_x2 = 10
R_x1 = 332  # 560
R_x2 = 422
road_width = R_x2 - L_x2

# 바꿀 Pixel 
Ax1 = 40  # 50
Ax2 = 200  # 470
Ay1 = 0
Ay2 = 432

pts1 = np.float32([[L_x1, y1], [R_x1, y1], [L_x2, y2], [R_x2, y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])
###########################################################################
'''
''' Rotation 한 후 
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
'''
M = cv2.getPerspectiveTransform(pts1, pts2)
i_M = cv2.getPerspectiveTransform(pts2, pts1)

real_Road_Width = 125

#########################################################################################
##################################Sub-Functions##########################################


def hsv_Image_Processing(img):
    #############가우시안 블러#############
    # #Gaussian Blur Filter 를 씌워 Noise 를 없앰.
    blur = cv2.GaussianBlur(img, (3, 3), 0)  # 여기 (3,3)은 kernel 값. 조절 가능(Only 홀수)
    cv2.imshow('Blur',blur)
    ####################################

    ############HSV로 바꾸기##############
    # BGR image 를 HSV image 로 변환하여 차선만 잘 보이게 함.
    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160])  # 이 Lower 값을 조절하여 날씨에 대한 대응 가능.
    upper = np.array([255, 255, 255])
    # Lower, Upper 에서 건드리는 건 hsv 중 v(Value)값임.[명도]
    mask = cv2.inRange(img_hsv, lower, upper)
    hsv = cv2.bitwise_and(blur, blur, mask=mask)
    cv2.imshow('hsv',hsv)
    #####################################

    ##############Canny Edge#############
    img_canny = cv2.Canny(hsv, 20, 80)
    return img_canny

def bgr_Image_Processing(img):
    ##############가우시안 블러#############
    blur = cv2.GaussianBlur(img, (3,3), 0)
    cv2.imshow('Blur',blur)
    #####################################

    #############Extract White############
    bgr_Threshold = [200, 200, 200]
    thresholds = (img[:, :, 0] < bgr_Threshold[0]) \
                 | (img[:, :, 1] < bgr_Threshold[1]) \
                 | (img[:, :, 2] < bgr_Threshold[2])
    mark[thresholds] = [0, 0, 0]

def set_Gray(img, region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, (255, 255, 255))
    img_ROI = cv2.bitwise_and(img, mask)
    return img_ROI

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
    global edge_lx, edge_rx

    # draw line roi
    cv2.polylines(dst, np.int32([L_line]), 1, (0, 255, 0), 5)
    cv2.polylines(dst, np.int32([R_line]), 1, (0, 255, 0), 5)
    cv2.imshow('ROI added', dst)  # --> ROI 부분만 추가됨.

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
    # cv2.imshow('Extract line fin', dst)

    return dst, edge_lx, edge_ly, edge_rx, edge_ry


def poly_Ransac(x_points, y_points, y_min, y_max):


    return

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

#Detect Stop Line
def detect_Stop(dst, dst_canny, L_roi, R_roi):
    stop_Roi = np.array([[(45, 440), (205, 440), (205, 5), (45, 5)]])
    # cv2.polylines(dst, stop_Roi, 1, (0,155,0),5)
    img_Stop = set_Gray(dst_canny, stop_Roi)
    line_arr = cv2.HoughLinesP(img_Stop, 1, 1 * np.pi / 180, 30, np.array([]), 10, 30)
    line_arr = np.array(np.squeeze(line_arr))
    line_arr_t = line_arr.transpose()
    if line_arr.shape != ():
        slope_Degree = ((np.arctan2(line_arr_t[1] - line_arr_t[3], line_arr_t[0] - line_arr_t[2]) * 180) / np.pi)
        try:
            line_arr = line_arr[np.abs(slope_Degree) > 160]
            line_arr = line_arr[:, None]
        except IndexError:
            pass
        else:
            stop_Lines = get_Fit_Line(line_arr)
            try:
                cv2.line(dst, (stop_Lines[0], stop_Lines[1]), (stop_Lines[2], stop_Lines[3]), (0, 155, 0), 5)
            except TypeError:
                pass
            return dst, stop_Lines


###############################Main Function#################################

def lane_Coef(img):
    #이차함수의 계수 3개를 보여줌.

    return coef1, coef2, coef3
    #이차함수 꼴 = coef1*(x^2) + coef2*x + coef3




#CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/Parking Detection.mp4'
CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/0507_one_lap_normal.mp4'
#CAM_ID = 1

cam = cv2.VideoCapture(CAM_ID)  # 카메라 생성
if cam.isOpened() == False:  # 카메라 생성 확인
    print('Can\'t open the CAM')


# 카메라 이미지 해상도 얻기
cam.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,270)

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
    #cv2.imshow('ROTATED',rotated)
    height, width = rotated.shape[:2]
    dst = cv2.warpPerspective(frame, M, (width, height))
    #dst = cv2.warpPerspective(rotated, M, (height, width)) 이건 Rotate 한 후의 코드.
    cv2.imshow('dst',dst)
    hsv_Image_Processing(dst)
    i_dst = cv2.warpPerspective(dst, i_M, (bird_height, bird_width))
    cv2.imshow('asd', i_dst)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)




