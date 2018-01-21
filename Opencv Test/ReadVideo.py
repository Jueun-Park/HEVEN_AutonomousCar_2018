###########ReadVideo###########
###Sample By JG

import numpy as np
import cv2

def ReadVideo():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture(1)
    except:
        print('카메라 구동 실패')
        return

    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()

        if not ret:
            print('비디오 읽기 오류')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('VIDEO', gray)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

ReadVideo()

    
        

