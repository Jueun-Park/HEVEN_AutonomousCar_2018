##########VideoBlurEdge##########



import numpy as np
import cv2


def VideoMorphEdge():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/project_video.mp4')
    except:
        print('카메라 구동 실패')
        return

    cap.set(3, 540)
    cap.set(4, 270)

    while True:
        ret, frame = cap.read()

        if not ret:
            print('비디오 읽기 오류')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('VIDEO', gray)

        kernel = np.ones((5, 5), np.float32)/25
        blur = cv2.filter2D(gray, -1, kernel)
        cv2.imshow('Blur', blur)

        edge = cv2.Canny(blur, 100, 150)
        cv2.imshow('Canny Edge', edge)



        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

VideoMorphEdge()