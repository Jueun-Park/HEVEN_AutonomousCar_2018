#########2CamsSaving#########
###Sample By JG

import cv2
import numpy as np

def WriteVideo():
    try:
        print('카메라를 구동합니다.')
        cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(2)
    except:
        print('카메라 구동 실패')
        return

    fps1 = 30.0
    width1 = int(cap1.get(3))
    height1 = int(cap1.get(4))
    fcc1 = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    out1 = cv2.VideoWriter('Cam1.avi', fcc1, fps1, (width1, height1))
    print('1 Cam 녹화를 시작합니다.')

    fps2 = 30.0
    width2 = int(cap2.get(3))
    height2 = int(cap2.get(4))
    fcc2 = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    out2 = cv2.VideoWriter('Cam2.avi', fcc2, fps2, (width2, height2))
    print('2 Cam 녹화를 시작합니다.')

    while True:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()


        if not ret:
            print('비디오 읽기 오류')
            break

        cv2.imshow('Video1', frame1)
        cv2.imshow('Video2', frame2)

        out1.write(frame1)
        out2.write(frame2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print('녹화를 종료합니다.')
            break

    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()


WriteVideo()
