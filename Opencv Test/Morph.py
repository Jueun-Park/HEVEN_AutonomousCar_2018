############Morph###########
'''
For Better Image, Erosion and Dilation Executes.
'''

import numpy as np
import cv2

def Morph():
    img = cv2.imread('C:/Users/jglee/Desktop/IMAGES/MONG.jpg', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3,3), np.uint8)

    erosion = cv2.erode(img, kernel, iterations= 1)
    dilation = cv2.dilate (img, kernel, iterations= 1)

    cv2.imshow('ORIGINAL', img)
    cv2.imshow('EROSION', erosion)
    cv2.imshow('DILATION',dilation)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

Morph()