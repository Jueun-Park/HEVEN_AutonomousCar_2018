import cv2
import numpy as np



def perspective_Hough():
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

        Cut_Frame = frame[560:680, 0:1200]
        if not ret:
            print('비디오 읽기 오류')
            break

        gray = cv2.cvtColor(Cut_Frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('VIDEO', gray)

        h, w = frame.shape[:2]

        pts1 = np.float32([[0,0], [1200,0], [0,600], [1200,600]])
        pts2 = np.float32([[0,0], [1200,0], [520,720], [680,720]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        img = cv2.warpPerspective(gray, M, (w,h))

        kernel = np.ones((5, 5), np.float32)/25
        blur = cv2.filter2D(img, -1, kernel)
        cv2.imshow('Blur', blur)

        edge = cv2.Canny(blur, 100, 150)
        cv2.imshow('Blur Canny Edge', edge)

        try:
            lines = cv2.HoughLines(edge, 1, np.pi / 180, 200)
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('houghlines3.jpg', img)

        except:
            cv2.imshow('houghlines3.jpg', img)



        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

perspective_Hough()