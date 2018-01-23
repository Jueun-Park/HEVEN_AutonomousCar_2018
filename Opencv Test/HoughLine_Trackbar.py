##########Hough Transform#############
###Sample by JG
import cv2
import numpy as np

def onChange(pos):
    global img
    global gray
    global dst

    dst = np.copy(img)

    apertureSize = cv2.getTrackbarPos("ApertureSize", "Result")
    minLineLength = cv2.getTrackbarPos("LineLength", "Result")
    maxLineGap = cv2.getTrackbarPos("LineGap", "Result")

    # according to OpenCV, aperture size must be odd and between 3 and 7
    if apertureSize % 2 == 0:
        apertureSize += 1
    if apertureSize < 3:
        apertureSize = 3

    edges = cv2.Canny(gray,100,200,apertureSize = apertureSize)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,50)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(dst,(x1,y1),(x2,y2),(255,255,255),2)

#Run Main
if __name__ == "__main__" :

    img = cv2.imread("C:/Users/jglee/Desktop/IMAGES/test3.jpg", -1)
    dst = np.copy(img)

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #default values for trackbars
    defaultApertureSize = 5
    minLineLength = 0
    maxLineGap = 19

    # according to OpenCV, aperture size must be odd and between 3 and 7
    # the aperture size range is (0 - 6)
    cv2.createTrackbar("ApertureSize", "Result", defaultApertureSize, 6, onChange)

    # line length range is (0 - 10)
    cv2.createTrackbar("LineLength", "Result", minLineLength, 10, onChange)

    # line gap range is (0 - 19)
    cv2.createTrackbar("LineGap", "Result", maxLineGap, 19, onChange)

    while True:
        cv2.imshow("Result", dst)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
