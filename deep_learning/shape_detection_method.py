import numpy as np
import cv2
import time

def shape_detect(img):
    sign = []

    if img is None:
        print("image is none")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img6 = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(img6, 90, 180, apertureSize=3)
        ret, thresh = cv2.threshold(edges, 127, 255, 0)
        image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:

            (x, y, w, h) = cv2.boundingRect(cnt)

            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            le = max(w, h) + 10

            if w > 60 and h > 60:
                x_1 = int(x + (w - le) / 2)
                x_2 = int(x + (w + le) / 2)
                y_1 = int(y + (h - le) / 2)
                y_2 = int(y + (h + le) / 2)

                img_trim = img[y_1: y_2, x_1:x_2]
                height, width = img_trim.shape[:2]

                if -5 < (height - width) < 5:
                    img_trim_resize = cv2.resize(img_trim, (32, 32),interpolation=cv2.INTER_AREA)
                    sign.append(img_trim_resize)
    return sign


def main():
    Shape_detection = shape_detect(img)
    print(Shape_detection)

if __name__ == "__main__":
    # open cam
    cam = cv2.VideoCapture(0)

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