import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import (LinearRegression, RANSACRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

######################################변수 설정######################################
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
height_ROI = 270 # 얘만 잘 조절하면 ROI 길이 조절 가능.
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
L_x1 = 160  # 400
L_x2 = 0
R_x1 = 320  # 560
R_x2 = 480
road_width = R_x2 - L_x2

# 바꿀 Pixel
Ax1 = 60
Ax2 = 210
Ay1 = 0
Ay2 = 480

# Homograpy transform
pts1 = np.float32([[L_x1, y1], [R_x1, y1], [L_x2, y2], [R_x2, y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])
M = cv2.getPerspectiveTransform(pts1, pts2)
i_M = cv2.getPerspectiveTransform(pts2, pts1)

real_Road_Width = 125


###################################Sub Functions####################################


# set ROI of gray scale
def set_Gray(img, region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, (255, 255, 255))
    img_ROI = cv2.bitwise_and(img, mask)
    return img_ROI


# set ROI of red scale
def set_Red(img, region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, (0, 0, 255))
    img_red = cv2.bitwise_and(img, mask)
    return img_red


# Linear ransac
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


# Polynomial ransac
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


# draw polynomial
def draw_Poly(img, points, color):
    for i in range(num_y):
        try:
            cv2.circle(img, (int(points[i]), height_ROI + i), 1, color, 2)
        except TypeError:
            pass


# black bye
def black_Bye(img, threshold):
    thresholds = (img[:, :, 2] < threshold)
    img[thresholds] = [0, 0, 0]
    return img


def con_Bye(img, th_green):
    thresholds = (img[:, :, 1] < th_green)

    img[thresholds] = [0, 0, 0]

    return img

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


# image processing
def image_Processing(img):
    blur = gaussian_Blur(img)
    hsv = BGR2HSV(blur)
    #cv2.imshow('hsv', hsv)
    img_canny = cv2.Canny(hsv, 20, 80)
    #cv2.imshow('Canny',img_canny)

    return img_canny


# choose roi
def choose_Roi(dst, direction, L_num, R_num, L_ransac, R_ransac, L_roi_before, R_roi_before):
    # left line roi
    try:
        if L_num != 0:
            if direction == 'left' or direction == 'right':
                L_roi = np.array([[(int(L_ransac[0]) - 25, height_ROI), (int(L_ransac[0]) + 25, height_ROI),
                                   (int(L_ransac[num_y // 3]) + 25, height_ROI + num_y // 3), (80, bird_height ),
                                   (20, bird_height ), (int(L_ransac[num_y // 3]) - 25, height_ROI + num_y // 3)]])
            else:
                L_roi = np.array([[(int(L_ransac[0]) - 25, height_ROI), (int(L_ransac[0]) + 25, height_ROI),
                                   (int(L_ransac[num_y // 3]) + 25, height_ROI + num_y // 3),
                                   (int(L_ransac[num_y - 50]) + 25, bird_height ),
                                   (int(L_ransac[num_y - 50]) - 25, bird_height ),
                                   (int(L_ransac[num_y // 3]) - 25, height_ROI + num_y //3)]])
        elif direction == 'straight':
            L_roi = np.array([[(0, 280), (bird_width / 2 - 40, 280), (bird_width / 2 - 40, height_ROI + num_y / 2),
                               (bird_width / 2 - 40, bird_height ), (15, bird_height )]])
        elif direction == 'right':
            L_roi = L_roi_before
        elif direction == 'left':
            L_roi = L_roi_before
        else:
            L_roi = L_roi_before

    except TypeError:
        L_roi = np.array([[(0, 280), (bird_width / 2 - 40, 280), (bird_width / 2 - 40, height_ROI + num_y / 2),
                           (bird_width / 2 - 40, bird_height ), (15, bird_height )]])

    # right line roi
    try:
        if R_num != 0:
            if direction == 'left' or direction == 'right':
                R_roi = np.array([[(250, bird_height ), (190, bird_height ),
                                   (int(R_ransac[num_y // 3]) - 25, height_ROI + num_y // 3),
                                   (int(R_ransac[0]) - 25, height_ROI),
                                   (int(R_ransac[0]) + 25, height_ROI),
                                   (int(R_ransac[num_y // 3]) + 25, height_ROI + num_y // 3)]])
            else:
                R_roi = np.array([[(int(R_ransac[num_y - 100]) + 25, bird_height ),
                                   (int(R_ransac[num_y - 50]) - 25, bird_height ),
                                   (int(R_ransac[num_y // 3]) - 25, height_ROI + num_y // 3),
                                   (int(R_ransac[0]) - 25, height_ROI),
                                   (int(R_ransac[0]) + 25, height_ROI),
                                   (int(R_ransac[num_y // 3]) + 25, height_ROI + num_y // 3)]])

        elif direction == 'straight':
            R_roi = np.array([[(bird_width - 15, bird_height ), (bird_width / 2 + 40, bird_height ),
                               (bird_width / 2 + 40, height_ROI + num_y / 2), (bird_width / 2 + 40, 280),
                               (bird_width, 280)]])

        elif direction == 'right':
            R_roi = R_roi_before

        elif direction == 'left':
            R_roi = R_roi_before
        else:
            R_roi = R_roi_before
    except TypeError:
        R_roi = np.array([[(bird_width - 15, bird_height ), (bird_width / 2 + 40, bird_height ),
                           (bird_width / 2 + 40, height_ROI + num_y / 2), (bird_width / 2 + 40, 280),
                           (bird_width, 280)]])
    return L_roi, R_roi


# decide left, right edge points
def extract_Line(dst, img_canny, L_line, R_line):
    global edge_lx, edge_rx
    # draw line roi
    cv2.polylines(dst, np.int32([L_line]), 1, (0, 255, 0), 5)
    cv2.polylines(dst, np.int32([R_line]), 1, (0, 255, 0), 5)
    #cv2.imshow('ROI added',dst) #--> ROI 부분만 추가됨.

    # canny edge
    L_edge = set_Gray(img_canny, np.int32([L_line]))
    R_edge = set_Gray(img_canny, np.int32([R_line]))

    # separate edge points
    edge_lx, edge_ly = np.where(L_edge >= 255)
    edge_rx, edge_ry = np.where(R_edge >= 255)


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
    #cv2.imshow('Extract line fin', dst)
    return dst, edge_lx, edge_ly, edge_rx, edge_ry


# check error
def check_Error(L_ransac, R_ransac, L_check, R_check, L_num, R_num, direction, road_Width):
    global mid_ransac

    # 4. Left Lane have to primary than Right Lane
    try:
        if (not R_E) and (L_ransac[0] > mid_ransac + 30 or L_ransac[209] > mid_ransac + 30):
            print("ERROR 4")
            L_ransac = copy.deepcopy(L_check)
            L_num = -1

    except TypeError:
        L_num = -1

    try:
        if (not L_E) and (R_ransac[0] < mid_ransac - 30 or R_ransac[209] < mid_ransac - 30):
            print("ERROR 4")
            R_ransac = copy.deepcopy(R_check)
            R_num = -1
    except TypeError:
        R_num = -1

    # 5. Left & Right lane should be in certain distance of virtual mid lane
    try:
        if abs(mid_ransac - L_ransac[0]) > 120:
            print("ERROR 5")
            L_ransac = copy.deepcopy(L_check)
            L_num = -1
    except TypeError:
        L_num = -1
    try:
        if abs(mid_ransac - R_ransac[0]) > 120:
            print("ERROR 5")
            R_ransac = copy.deepcopy(R_check)
            R_num = -1
    except TypeError:
        R_num = -1

    return L_ransac, R_ransac, L_num, R_num


# reset error 3 frame
def error_3frames(L_num, R_num, L_error, R_error, start_num):
    global L_E, R_E
    L_E = 0
    R_E = 0
    if L_num == -1:
        L_E = 1
        if L_error == 0 or L_error == 1 or L_error == 2:
            L_error += 1
        else:
            L_error = 0
            start_num = -1  # roi
    else:
        L_error = 0

    if R_num == -1:
        R_E = 1
        if R_error == 0 or R_error == 1 or R_error == 2:
            R_error += 1
        else:
            R_error = 0
            start_num = -1
    else:
        R_error = 0
    return L_error, R_error, start_num


# check road width
def check_Road_Width(L_ransac, R_ransac):
    road_Width = [0, 0, 0]
    try:
        road_Width[0] = R_ransac[0] - L_ransac[0]
        road_Width[1] = R_ransac[num_y // 3] - L_ransac[num_y // 3]
        road_Width[2] = R_ransac[num_y - 1] - L_ransac[num_y - 1]
    except TypeError:
        return 0
    return max(road_Width)


# check direction
def check_Direction(L_ransac, R_ransac, direction_before):
    try:
        L_dif = L_ransac[0] - L_ransac[num_y // 3]
        R_dif = R_ransac[0] - R_ransac[num_y // 3]
    except TypeError:
        direction = 'straight'
    else:
        if direction_before == 'right':
            if L_dif < 30 and R_dif < 30:
                #print ('str')
                direction = 'str'
            else:
                direction = 'right'
                #print ('right')
        elif direction_before == 'left':
            if L_dif > -30 and R_dif > -30:
                #print ('str')
                direction = 'str'
            else:
                direction = 'left'
                #print ('left')
        else:
            if L_dif > 45 and R_dif > 15:
                #print ('right')
                direction = 'right'
            elif R_dif < -45 and L_dif < -15:
                #print ('left')
                direction = 'left'
            else:
                direction = 'straight'
                #print ('str')
    return direction


# draw straight line
def draw_Straight_Line(dst, L_points, R_points, L_check, R_check, L_num, R_num, L_color, R_color):
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
def draw_Poly_Line(dst, L_points, R_points, L_check, R_check, L_num, R_num, L_color, R_color):
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


# detect stop line
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

#Rotation Function added(2018)

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


#####################################Main Function###################################


def lane_Detection(img):
    global direction, L_num, R_num, L_ransac, R_ransac, L_roi, R_roi, start_num, L_error, R_error
    global frame_num, L_check, R_check, stop_Lines, destination_J, destination_I
    global mid_ransac

    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.warpPerspective(img, M, (height, width))
    cv2.imshow('d',dst)

    img_canny = image_Processing(dst)

    L_roi, R_roi = choose_Roi(dst, direction, L_num, R_num, L_ransac, R_ransac, L_roi, R_roi)
    dst, edge_lx, edge_ly, edge_rx, edge_ry = extract_Line(dst, img_canny, L_roi, R_roi)
    #cv2.imshow('extract',dst)
    L_ransac = polynomial_Ransac(edge_ly, edge_lx, height_ROI, bird_height)
    R_ransac = polynomial_Ransac(edge_ry, edge_rx, height_ROI, bird_height)

    L_linear = linear_Ransac(edge_ly, edge_lx, height_ROI, bird_height)
    R_linear = linear_Ransac(edge_ry, edge_rx, height_ROI, bird_height)
    road_Width = check_Road_Width(L_ransac, R_ransac)

    if start_num == 0:
        L_check = copy.deepcopy(L_ransac)
        R_check = copy.deepcopy(R_ransac)

    direction = check_Direction(L_ransac, R_ransac, direction)
    if direction == 'straight':
        L_linear, R_linear, L_num, R_num = check_Error(L_linear, R_linear, L_check, R_check, L_num, R_num, direction,
                                                       road_Width)
        L_error, R_error, start_num = error_3frames(L_num, R_num, L_error, R_error, start_num)
        draw_Poly_Line(dst, L_ransac, R_ransac, L_check, R_check, L_num, R_num, (0, 255, 255), (255, 0, 255))
        #draw_Straight_Line(dst, L_linear, R_linear, L_check, R_check, L_num, R_num, (0, 0, 255), (255, 0, 0))
        #cv2.imshow('asdasdasd',dst)
        L_check = copy.deepcopy(L_linear)
        R_check = copy.deepcopy(R_linear)
        try:
            dst, stop_Lines = detect_Stop(dst, img_canny, L_roi, R_roi)
        except TypeError:
            pass
    else:
        L_ransac, R_ransac, L_num, R_num = check_Error(L_ransac, R_ransac, L_check, R_check, L_num, R_num, direction,
                                                       real_Road_Width)
        L_error, R_error, start_num = error_3frames(L_num, R_num, L_error, R_error, start_num)

        L_check = copy.deepcopy(L_ransac)
        R_check = copy.deepcopy(R_ransac)
    cv2.imshow('dst', dst)
    rotated = Rotate(dst, 270)
    #cv2.imshow('Rotated', rotated)
    i_dst = cv2.warpPerspective(dst, i_M, (bird_height, bird_width))
    cv2.imshow('asd',i_dst)
    start_num += 1
    frame_num += 1
    L_num += 1
    R_num += 1
    print('start num:',start_num)
    print('frame num:',frame_num)
    print('L num',L_num)
    print('R num',R_num)
    return stop_Lines




cam = cv2.VideoCapture('C:/Users/zenon/Desktop/0507_one_lap_normal.mp4')
#cam = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/Parking Detection.mp4')
#cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,270)

w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('size = ', w, h)

if (not cam.isOpened()):
    print ("cam open failed")

while True:
    s, img = cam.read()
    s_Lines = lane_Detection(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
