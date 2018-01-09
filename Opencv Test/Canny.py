
import numpy as np
import cv2

def canny(Input_Image):
    img = cv2.imread(Input_Image, cv2.IMREAD_GRAYSCALE)

    edge = cv2.Canny(img, 100, 200)

    cv2.imshow('Canny Edge', edge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


