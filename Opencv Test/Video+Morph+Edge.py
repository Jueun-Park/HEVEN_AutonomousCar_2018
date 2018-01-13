##########VideoMorphEdge#############3
'''

Opening executes well to identifying lane

'''

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

        kernel = np.ones((5, 5), np.uint8)

        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)

        edge1 = cv2.Canny(frame, 100, 150)
        cv2.imshow('Canny Edge1', edge1)

        edge2 = cv2.Canny(opening, 100, 150)
        cv2.imshow('Canny Edge2', edge2)

        edge3 = cv2.Canny(closing, 100, 150)
        cv2.imshow('Canny Edge3', edge3)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

VideoMorphEdge()
