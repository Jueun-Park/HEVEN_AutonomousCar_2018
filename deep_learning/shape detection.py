import numpy as np
import cv2

cap = cv2.VideoCapture(1)
if not cap.read():
    print("none")


if cap.read():
    count_2 = 0
    while True:
        count_2 +=1
        ret, img = cap.read()
        #       frame = cv2.resize(frame, (640,480))
        if ret:
            cv2.imshow('image', img)
        #       img = cv2.imread('sky.jpg')
        else:
            print("no frame")
            break
        if img is None:
            print("image is none")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            ret, thresh = cv2.threshold(gray, 127, 255, 1)

            image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                name = './images/new.jpg' + str(count)+ '_'+str(count_2)+ '.jpg'
                (x, y, w, h) = cv2.boundingRect(cnt)

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                le = max(w,h)

                stad = [64, 128, 192, 256, 320, 384, 448]

                for i in stad:
                    if i-64<le<i:
                        x_1 = int (x+(w-i)/2)
                        x_2 = int (x+(w+i)/2)
                        y_1 = int (y+(h-i)/2)
                        y_2 = int (y+(h+i)/2)

                        img_trim = img[y_1 : y_2,x_1 :x_2]

                        height, width = img_trim.shape[:2]
                        image = cv2.resize(img_trim, (width*32/i,height*32/i),interpolation=cv2.INTER_AREA)

                cv2.imwrite(name,image)

                count += 1
        cv2.imshow('img', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("fail")