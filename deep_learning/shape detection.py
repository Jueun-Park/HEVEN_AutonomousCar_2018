import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 448)

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
            img5 = cv2.GaussianBlur(img, (5, 5), 0)
            gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 129, 194)
            image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                le = max(w,h)+10

                if w>40 and h>40:
                    x_1 = int (x+(w-le)/2 -5)
                    x_2 = int (x+(w+le)/2 +5)
                    y_1 = int (y+(h-le)/2 -5)
                    y_2 = int (y+(h+le)/2 +5)
                    img_trim = img[y_1 : y_2 , x_1 :x_2]
                    height, width = img_trim.shape[:2]
                    limit = height - width

                    if limit > -5 and limit < 5:
                        if x_1 >0 and y_1 > 0:
                            cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (255, 0, 0), 4)
                            name = "./images/" + str(count) + "_" + str(count_2) + "_"+ str(height) +".png"
                            img_trim = cv2.resize(img_trim, (32,32))
                            cv2.imwrite(name,img_trim)
                count += 1

        cv2.imshow('img', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("fail")