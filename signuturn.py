import sign_cam2
import cv2
import os


def sign_camstart():
    cam = cv2.VideoCapture('C:/Users/Administrator/Desktop/0507_one_lap_normal.mp4')

    cam.set(3, 480)
    cam.set(4, 270)

    if (not cam.isOpened()):
        print("cam open failed")
    print(os.getpid())
    while True:
        s, img = cam.read()

        sign_cam2.u_turn_detect(img)

        cv2.imshow('camuturn', img)
        if cv2.waitKey(30) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)




