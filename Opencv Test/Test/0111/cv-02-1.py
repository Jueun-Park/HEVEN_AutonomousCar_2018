import numpy as np
import cv2

def tracking():
    try:
        print('카메라를 구동합니다.')
        cap=cv2.VideoCapture(0)
    except:
        print('카메라 구동 실패')
        return

    while True:
        ret, frame = cap.read()

        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        min_blue = np.array([75,100,100])
        max_blue = np.array([130,255,255])
        min_green = np.array([30,100,100])
        max_green = np.array([90,255,255])
        min_red = np.array([160,100,100])
        max_red = np.array([179,255,255])

        mask_blue=cv2.inRange(hsv,min_blue,max_blue)
        mask_green=cv2.inRange(hsv,min_green,max_green)
        mask_red=cv2.inRange(hsv,min_red,max_red)

        res1 = cv2.bitwise_and(frame,frame,mask=mask_blue)
        res2 = cv2.bitwise_and(frame,frame,mask=mask_green)
        res3 = cv2.bitwise_and(frame,frame,mask=mask_red)

        cv2.imshow('original',frame)
        cv2.imshow('BLUE',res1)
        cv2.imshow('GREEN',res2)
        cv2.imshow('RED',res3)

        k=cv2.waitKey(1) & 0xFF
        if k==27:
            break
        
    cv2.destroyAllWindows()

tracking()
