import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,270)

w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('size = ', w, h)

if (not cam.isOpened()):
    print ("cam open failed")

while True:
    s, img = cam.read()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 160])  # 이 Lower 값을 조절하여 날씨에 대한 대응 가능.
    upper = np.array([255, 255, 255])
    # Lower, Upper 에서 건드리는 건 hsv 중 v(Value)값임.[명도]
    mask = cv2.inRange(img_hsv, lower, upper)
    hsv = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('hsv_Cvt',hsv)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)