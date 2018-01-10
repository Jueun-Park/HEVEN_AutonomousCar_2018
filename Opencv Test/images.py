

import numpy as np
import cv2
import matplotlib.pyplot as plt


def hough(thr):
    img = cv2.imread('C:/Users/jglee/Desktop/IMAGES/KATARI2_480_270.mp4_20180105_142935.309.jpg')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(imgray, 150, 250, apertureSize = 3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, thr)

    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imshow('WHAT?', edges)
    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hough(100)
