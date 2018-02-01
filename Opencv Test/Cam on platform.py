import cv2
import numpy as np

#저번대회의 코드
#(Rotation 을 적용하지 않은 경우)
# set cross point (Rotation 때문에 저번대회랑 다름)
y1 = 160
y2 = 239

# 원래 Pixel (Rotation 때문에 저번대회랑 다름)
L_x1 = 100  # 400
L_x2 = 10
R_x1 = 332  # 560
R_x2 = 422
road_width = R_x2 - L_x2

# 바꿀 Pixel (Rotation 때문에 저번대회랑 다름)
Ax1 = 40  # 50
Ax2 = 200  # 470
Ay1 = 0
Ay2 = 432

pts1 = np.float32([[L_x1, y1], [R_x1, y1], [L_x2, y2], [R_x2, y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax2, Ay1], [Ax1, Ay2], [Ax2, Ay2]])


''' Rotation 한 후 
# set cross point (Rotation 때문에 저번대회랑 다름)
x1 = 185
x2 = 269

# 원래 Pixel (Rotation 때문에 저번대회랑 다름)
L_y1 = 320
L_y2 = 479
R_y1 = 160
R_y2 = 1
road_width = R_y2 - L_y2

# 바꿀 Pixel (Rotation 때문에 저번대회랑 다름)
Ax1 = 0
Ax2 = 480
Ay1 = 210
Ay2 = 60

# Homograpy transform
pts1 = np.float32([[x1, L_y1], [x1, R_y1], [x2, R_y2], [x2, L_y2]])
pts2 = np.float32([[Ax1, Ay1], [Ax1, Ay2], [Ax2, Ay2], [Ax2, Ay1]])
#pts1 = np.float32([[185, 320], [185, 160], [269, 1], [269, 479]])
#pts2 = np.float32([[0, 210], [0, 60], [480, 60], [480, 210]])
'''
M = cv2.getPerspectiveTransform(pts1, pts2)
i_M = cv2.getPerspectiveTransform(pts2, pts1)

real_Road_Width = 125


#cam = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/0507_one_lap_normal.mp4')
#cam = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/Parking Detection.mp4')
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
    height, width = img.shape[:2]
    cv2.imshow('Original',img)
    dst = cv2.warpPerspective(img, M, (height, width))
    cv2.imshow('d', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)

