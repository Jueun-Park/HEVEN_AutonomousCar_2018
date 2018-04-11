import numpy as np
import cv2

cap = cv2.VideoCapture(0)
if not cap.read():
    print("none")

if cap.read():
    while True:
        ret, img = cap.read()
#       frame = cv2.resize(frame, (640,480))
        if ret:
            cv2.imshow('image',img)
#       img = cv2.imread('sky.jpg')
        else:
            print("no frame")
            break
        if img is None:
            print("image is none")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            ret,thresh = cv2.threshold(gray,127,255,1)

            image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for cnt in contours:
                name='./new.jpg'+str(count)+'.jpg'
                (x,y,w,h) = cv2.boundingRect(cnt)
#               cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                if w<128 and h<128:
                    img_trim = img[y+int(h/2)-64:y+int(h/2)+64,x+int(h/2)-64:x+int(h/2)+64]
                    if img_trim is None:
                        print("trim is none")
                    else :
                        img_trim= cv2.resize(img_trim, None, fx=0.0625, fy=0.0625, interpolation=cv2.INTER_AREA)
                elif w<256 and h<256:
                    img_trim = img[y+int(h/2)-128:y+int(h/2)+128,x+int(h/2)-128:x+int(h/2)+128]
                    if img_trim is None:
                        print("trim is none")
                    else :
                        img_trim=cv2.resize(img_trim, None, fx=0.0625, fy=0.0625, interpolation=cv2.INTER_AREA)
                else :
                    img_trim = img[y+int(h/2)-240:y+int(h/2)+240,x+int(h/2)-240:x +int(h/2)+240]
                    if img_trim is None:
                        print("trim is none")
                    else :
                        img_trim=cv2.resize(img_trim, None, fx=0.0625, fy=0.0625, interpolation=cv2.INTER_AREA)
                cv2.imwrite(name,img_trim)

                count+=1

        cv2.imshow('img', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print ("fail")