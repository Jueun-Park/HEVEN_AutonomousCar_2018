# -*- Encoding:UTF-8 -*- #
'''

2018/01/17 에 만든 Code
Video 에서 Bird-Eye-View 를 적용 후 Gaussian Filter & Opening Filter 적용
이후 Canny Edge Detection을 이용해 Edge를 추출 후 Houghline 그리기.

'''

import cv2
import numpy as np


def lane_Detection():

    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/project_video.mp4')
    except:
        print('카메라 구동 실패')
        return
    #나중에 Cam 을 실행하게 되면 크기를 결정. 지금은 640x480
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    while True:
        ret, frame = cap.read()

        if not ret:
            print('비디오 읽기 오류')
            break

        #GRAYSCALE로 바꾼 함수.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Output',gray)

        ROI = gray[480:720, 0:1200]
        cv2.imshow('ROI',ROI)

        #Bird_View에 대한 함수.
        h, w = ROI.shape[:2]

        pts1 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
        pts2 = np.float32([[0, 0], [1280, 0], [480, 720], [800, 720]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        img = cv2.warpPerspective(ROI, M, (w, h))

        cv2.imshow('Bird_View', img)


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

lane_Detection()