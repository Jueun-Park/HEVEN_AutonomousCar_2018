import numpy as np
import cv2


import numpy as np
import cv2
import time

capture = cv2.VideoCapture(1)
capture2 = cv2.VideoCapture(0)
capture3 = cv2.VideoCapture(2)

while True:
    ret, frame = capture.read()
    ret2, frame2 = capture2.read()
    ret3, frame3 = capture3.read()

    cv2.imshow('webcam1', frame)
    cv2.imshow('webcam2', frame2)
    cv2.imshow('webcam3', frame3)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

capture.release()
capture2.release()

cv2.destroyAllWindows()