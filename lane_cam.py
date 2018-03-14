import numpy as np
import cv2

X = 10
Y = 10

# webcam open
camera_L = cv2.VideoCapture(0)
camera_R = cv2.VideoCapture(3)

# 양쪽 webcam 모두 해상도를 800x448 로 설정
camera_L.set(3, 800)
camera_L.set(4, 448)
camera_R.set(3, 800)
camera_R.set(4, 448)

'''
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output1 = cv2.VideoWriter('output_left3.avi', fourcc, 30.0, (640, 480))
output2 = cv2.VideoWriter('output_right.avi', fourcc, 30.0, (640, 480))
'''

# webcam 왜곡 보정에 필요한 값들(camera calibration 작업을 통해서 구했음)
camera_matrix_L = np.array([[474.383699, 0, 404.369647], [0, 478.128447, 212.932297], [0, 0, 1]])
camera_matrix_R = np.array([[473.334870, 0, 386.312394], [0, 476.881433, 201.662339], [0, 0, 1]])
distortion_coeffs_L = np.array([0.164159, -0.193892, -0.002730, -0.001859])
distortion_coeffs_R = np.array([0.116554, -0.155379, -0.001045, -0.001512])


while True:
    # 양쪽 webcam 으로부터 frame 읽어오기(ret 은 frame 이 정상적으로 읽혔는지를 나타내는 boolean 값)
    ret_L, frame_L = camera_L.read()
    ret_R, frame_R = camera_R.read()

    dst_L = cv2.undistort(frame_L, camera_matrix_L, distortion_coeffs_L, None, None)  # 왜곡 보정
    dst_L = dst_L[Y:448 - Y, X:800 - X]  # 이미지 crop (왜곡 보정 결과물의 테두리부분이 부자연스러워서 잘라버림)

    dst_R = cv2.undistort(frame_R, camera_matrix_R, distortion_coeffs_R, None, None)
    dst_R = dst_R[Y:448 - Y, X:800 - X]

    cv2.imshow('left_cam', dst_L)
    cv2.imshow('right_cam', dst_R)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera_L.release()
camera_R.release()

cv2.destroyAllWindows()