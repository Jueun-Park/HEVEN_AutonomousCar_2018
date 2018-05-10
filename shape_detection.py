# 표지판이 있는 위치를 자른 이미지 반환
# 이아영, 김윤진
# input: 캠 이미지
# output: 표지판 후보 이미지

import cv2
global count
import time

def shape_detect(img):
    sign = []

    if img is None:
        print("image is none")
    else:
        img5 = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 129, 194)
        image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            le = max(w, h) + 10

            if w > 40 and h > 40:
                x_1 = int(x + (w - le) / 2 - 5)
                x_2 = int(x + (w + le) / 2 + 5)
                y_1 = int(y + (h - le) / 2 - 5)
                y_2 = int(y + (h + le) / 2 + 5)
                img_trim = img[y_1: y_2, x_1:x_2]
                height, width = img_trim.shape[:2]
                limit = height - width

                if -5 < (height - width) < 5:
                    if x_1 > 0 and y_1 > 0:
                        cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (255, 0, 0), 4)
                        img_trim_resize = cv2.resize(img_trim, (32, 32))
                        sign.append(img_trim_resize)
    return sign


def main():
    Shape_detection = shape_detect(img)
    print(Shape_detection)


if __name__ == "__main__":
    # open cam
    cam = cv2.VideoCapture(2)
    count = 0
    if not cam.isOpened():
        print("cam open failed")
    while True:
        s, img = cam.read()
        main()
        count += 1
        cv2.imshow('cam', img)

        if cv2.waitKey(30) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)