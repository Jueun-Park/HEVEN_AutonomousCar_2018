import numpy as np
import cv2

cap = cv2.VideoCapture("C:/Users/LG/PycharmProjects/untitled6/previous_double_bend.mp4")
cap.set(3, 800)
cap.set(4, 448)

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
            image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            edges_copy = edges
            count = 0

            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                le = max(w,h)+10

                if w>60 and h>60:

                    x_1 = int (x+(w-le)/2)
                    x_2 = int (x+(w+le)/2)
                    y_1 = int (y+(h-le)/2)
                    y_2 = int (y+(h+le)/2)

                    cv2.rectangle(edges_copy, (x_1, y_1), (x_2, y_2), (255, 0, 0), 5)
                    img_trim = img[y_1 : y_2 , x_1 :x_2]
                    height, width = img_trim.shape[:2]
                    limit = height - width

                    if limit > -5 and limit < 5:
                        if x_1 >0 and y_1 > 0:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            img_trim = cv2.resize(img_trim, (32,32))

        cv2.imshow('image', edges)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("fail")