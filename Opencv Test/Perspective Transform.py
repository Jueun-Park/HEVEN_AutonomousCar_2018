import cv2
import numpy as np

def perspective_Transform():

    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/project_video.mp4')
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

        h, w = frame.shape[:2]

        pts1 = np.float32([[0,0], [1280,0], [0,720], [1280,720]])
        pts2 = np.float32([[0,0], [1280,0], [320,720], [960,720]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        img = cv2.warpPerspective(frame, M, (w,h))

        cv2.imshow('img', img)


        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

perspective_Transform()