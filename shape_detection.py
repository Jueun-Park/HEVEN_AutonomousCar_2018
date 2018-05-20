# 표지판이 있는 위치를 자른 이미지 반환
# 이아영, 김윤진
# input: 캠 이미지
# output: 표지판 후보 이미지

import cv2
import numpy as np

Y2_UPPER = 290
Y1_UPPER = 185
Y1_LOWER = 80

count = 0

# lower_yellow = np.array([0, 30, 40], np.uint8)
# upper_yellow = np.array([55, 197, 255], np.uint8)
#
# lower_blue = np.array([43, 5, 0], np.uint8)
# upper_blue = np.array([170, 90, 50], np.uint8)


lower_yellow = np.array([0, 0, 0], np.uint8)
upper_yellow = np.array([255, 255, 255], np.uint8)

lower_blue = np.array([0, 0, 0], np.uint8)
upper_blue = np.array([255, 255, 255], np.uint8)


def shape_detect(img):
    sign = []

    if img is None:
        print("image is none")
    else:
        img2 = img
        #cv2.line(img, (0, Y2_UPPER), (799, Y2_UPPER), (0, 0, 255), 2)
        #cv2.line(img, (0, Y1_UPPER), (799, Y1_UPPER), (0, 255, 0), 2)
        #cv2.line(img, (0, Y1_LOWER), (799, Y1_LOWER), (255, 0, 0), 2)
        img5 = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 52, 104, apertureSize=3)
        image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            le = max(w, h) + 10

            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = int(100 * area / hull_area)
                if solidity > 0 and w > 42 and h > 42:
                    x_1 = int(x + (w - le) / 2)
                    x_2 = int(x + (w + le) / 2)
                    y_1 = int(y + (h - le) / 2)
                    y_2 = int(y + (h + le) / 2)

                    if x_1 > 0 and Y2_UPPER > y_2 and Y1_UPPER > y_1 > Y1_LOWER:
                        # if x_1 > 0 and y_1 > 0:
                        img_trim = img2[y_1: y_2, x_1:x_2]

                        img_trim_resize = cv2.resize(img_trim, (32, 32))

                        yellow_filtered = cv2.inRange(img_trim_resize, lower_yellow, upper_yellow)
                        blue_filtered = cv2.inRange(img_trim_resize, lower_blue, upper_blue)
                        both = cv2.bitwise_or(yellow_filtered, blue_filtered)
                        cv2.imshow('filtered', both)
                        nonzero_num = np.count_nonzero(both != 0)

                        if nonzero_num > 200:
                            if (le > 60 and len(cnt) > 120) or (le > 60 and 60 < len(cnt) < 120) or len(cnt) < 60:
                                cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (255, 0, 0), 4)
                                sign.append(img_trim_resize)
                                name = "./images/" + str(solidity) + "_" + str(y_1) + "_" + str(y_2) + "_.png"
                                cv2.imwrite(name, img_trim)
    return sign


def main():
    Shape_detection = shape_detect(img)
    print(Shape_detection)


if __name__ == "__main__":
    # open cam
    cam = cv2.VideoCapture('C:/Users/Administrator/Documents/GOMCam/parking.mp4')
    cam.set(3, 800)
    cam.set(4, 448)

    if not cam.isOpened():
        print("cam open failed")
    while True:
        s, img = cam.read()
        main()
        cv2.imshow('cam', img)

        if cv2.waitKey(30) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
