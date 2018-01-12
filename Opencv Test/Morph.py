############Morph###########
'''
For Better Image, Erosion and Dilation Executes.
'''

import numpy as np
import cv2

def Morph():
    img = cv2.imread('C:/Users/jglee/Desktop/IMAGES/KATARI2_480_270.mp4_20180105_142941.742.jpg', cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((2,2), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

Morph()