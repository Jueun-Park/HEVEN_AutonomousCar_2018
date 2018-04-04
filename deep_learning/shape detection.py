import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.read():
    print("none")


if cap.read():
    count_2 = 0
    while True:
        time.sleep(0.3)
        count_2 +=1
        ret, img = cap.read()

        if img is None:
            print("image is none")
        else:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (7, 7), 0)
            # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

            # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
            # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,1,11,2)
            # ret, thresh = cv2.threshold(blur, 127, 255, 1)

            # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # img5 = cv2.GaussianBlur(img, (5, 5), 0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img6 = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(img6, 100, 180, apertureSize=3)
            ret, thresh = cv2.threshold(edges, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for cnt in contours:
                name = './images/new' + str(count)+ '_'+str(count_2)+ '.jpg'

                (x, y, w, h) = cv2.boundingRect(cnt)

#                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                le = max(w,h)+10

                if w>120 and h>120:
                    x_1 = int (x+(w-le)/2)
                    x_2 = int (x+(w+le)/2)
                    y_1 = int (y+(h-le)/2)
                    y_2 = int (y+(h+le)/2)

                    img_trim = img[y_1 : y_2 , x_1 :x_2]
                    height, width = img_trim.shape[:2]

                    if -5< (height - width) <5:
                        img_trim = cv2.resize(img_trim, (32,32))
                        cv2.imwrite(name,img_trim)

                count += 1
        cv2.imshow('img', thresh)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("fail")