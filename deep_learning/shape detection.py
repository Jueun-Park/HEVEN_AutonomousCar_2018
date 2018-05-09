import numpy as np
import cv2
import time

cap = cv2.VideoCapture('C:\\Users\Administrator\PycharmProjects\Lane_logging\sign_logging.avi')


if not cap.read():
    print("none")


if cap.read():
    count_2 = 0
    while True:
        # time.sleep(2)
        count_2 +=1
        ret, img = cap.read()

        if img is None:
            print("image is none")
        else:

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img6 = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(img6, 60, 180, apertureSize=3)
            ret, thresh = cv2.threshold(edges, 127, 255, 0)
            image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for cnt in contours:
                # name = "./image_files/SIZE/image" + str(count) + "_"+str(count_2)+ ".png"
                # name = "image.png"

                (x, y, w, h) = cv2.boundingRect(cnt)

                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                le = max(w,h)

                if w>30 and h>30:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    x_1 = int (x+(w-le)/2 -5)
                    x_2 = int (x+(w+le)/2 +5)
                    y_1 = int (y+(h-le)/2 -5)
                    y_2 = int (y+(h+le)/2 +5)

                    img_trim = img[y_1 : y_2 , x_1 :x_2]
                    height, width = img_trim.shape[:2]
                    limit = height - width

                    if limit > -5 and limit < 5:
                        if x_1 >0 and y_1 > 0:
                            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            name = "./image_files/SIZE/image" + str(height)+"_"+ str(count) + "_" + str(count_2) + ".png"
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