import numpy as np
import cv2

img = cv2.imread('./images/signs11.png')
if img is None:
    print("image is none")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,1)
#    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                                cv2.THRESH_BINARY, 15, 2)

    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        print (cnt)
        print (len(approx))
        if len(approx)==3:
            print ("triangle")
            cv2.drawContours(img,[cnt],-1,(0,255,0),2)
        elif len(approx)==4:
            print ("square")
            cv2.drawContours(img,[cnt],0,(0,0,255),3)
        elif len(approx) > 15:
            print ("circle")
            cv2.drawContours(img,[cnt],0,(0,255,255),3)


        leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
        topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

        # 좌표 표시하기
        cv2.circle(img, leftmost, 20, (0, 0, 255), -1)
        cv2.circle(img, rightmost, 20, (0, 0, 255), -1)
        cv2.circle(img, topmost, 20, (0, 0, 255), -1)
        cv2.circle(img, bottommost, 20, (0, 0, 255), -1)


    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
