import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow('image')

cv2.createTrackbar('threshold1', 'image', 0, 255, nothing)
cv2.createTrackbar('threshold2', 'image', 0, 255, nothing)

if not cap.read():
    print("none")

if cap.read():
    count_2 = 0
    while True:
        count_2 +=1
        ret, img = cap.read()

        if img is None:
            print("image is none")
        else:
            threshold1 = cv2.getTrackbarPos('threshold1', 'image')
            threshold2 = cv2.getTrackbarPos('threshold2', 'image')
            img5 = cv2.GaussianBlur(img, (5, 5), 0)
            gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)

        cv2.imshow('image', edges)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("fail")
