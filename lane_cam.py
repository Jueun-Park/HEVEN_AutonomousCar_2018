# -*- Encoding:UTF-8 -*- #

# 카메라 통신 및 차선 인식
# input: sign_cam
# output: 이차함수의 계수.

import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import (LinearRegression, RANSACRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline




# CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/Parking Detection.mp4'
CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/0507_one_lap_normal.mp4'
# CAM_ID = 1

cam = cv2.VideoCapture(CAM_ID)

if (not cam.isOpened()):
    print ("cam open failed")

def main(cam):

    while True:
        ret, frame = cam.read()
        cv2.imshow('asd',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main(cam)

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)