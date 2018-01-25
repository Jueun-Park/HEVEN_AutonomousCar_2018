import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calib, undistort
from threshold import gradient_combine, hls_combine, comb_result
from finding_lines import Line, warp_image, find_LR_lines, draw_lane, print_road_status, print_road_map



th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)


'''
c_rows, c_cols = combined_result.shape[:2]
print(c_rows, c_cols)

s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])
warp_img, M, Minv = warp_image(combined_result, src, dst, (720, 720))
cv2.imshow('warp', warp_img)

cv2.imshow('CG',combined_gradient)
cv2.imshow('CH',combined_hls)
cv2.imshow('CR',combined_result)
'''




cam = cv2.VideoCapture('C:/Users/jglee/Desktop/VIDEOS/project_video.mp4')

width = 480
height = 270

if (not cam.isOpened()):
    print ("cam open failed")

while True:
    ret, frame = cam.read()
    combined_gradient = gradient_combine(frame, th_sobelx, th_sobely, th_mag, th_dir)
    combined_hls = hls_combine(frame, th_h, th_l, th_s)
    combined_result = comb_result(combined_gradient, combined_hls)
    cv2.imshow('CR', combined_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)