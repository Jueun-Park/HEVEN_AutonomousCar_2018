import numpy as np
import cv2

fname = 'uturn.png'
img = cv2.imread(fname, cv2.IMREAD_COLOR)

img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # BGR format으로 되어 있는 사진을 HSV format으로 변환
img_hsv2 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # img_hsv와 같지만, 이 이미지에서 밝기를 조정할 예정

#img_h, img_s, img_v = cv2.split(img_hsv)

#cv2.imshow('img', img) # 원본 이미지 출력

height, width, channel = img.shape
#print (height, width, channel)


for i in range(0,5):
    for y in range(0, height):
        for x in range(0, width):
            # h = img_hsv.item(y,x,0)
            # s = img_hsv.item(y,x,1)
            v = img_hsv.item(y, x, 2) # HSV format에서 v 부분을 가져옴. (명도)
            img_hsv2.itemset((y, x, 2), max(0, min(v - 100 + 50*i, 255))) # 원래 사진의 v 부분에 대하여 밝기를 조정함. (0,255)

    img_final = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2BGR) # HSV format을 다시 BGR format으로 변환함.
    # cv2.imshow('final', img_hsv)
    filename = str(i) + '.png'
    print(filename)
    cv2.imwrite(filename, img_final) # 밝기가 조정된 사진을 저장함.


cv2.waitKey(0)
cv2.destroyAllWindows()