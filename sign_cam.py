# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)

# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
#          'MOVING_OBS': 3,'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

import cv2
import numpy as np
import time

# Sign Boards
try:
    parking_cascade = cv2.CascadeClassifier('./sign_xml_files/parkingdetect.xml')  # 1. 자동 주차
    static_cascade = cv2.CascadeClassifier('./sign_xml_files/static_0514.xml')  # 2. 정적 장애물
    moving_cascade = cv2.CascadeClassifier('./sign_xml_files/moving_0510.xml')  # 3. 동적 장애물
    s_curve_cascade = cv2.CascadeClassifier('./sign_xml_files/scurve_0517.xml')  # 4. S자 주행
    narrow_cascade = cv2.CascadeClassifier('./sign_xml_files/narrowno.xml')  # 5. 협로 주행
    u_turn_cascade = cv2.CascadeClassifier('./sign_xml_files/uturndetect.xml')  # 6. 유턴
    crosswalk_cascade = cv2.CascadeClassifier('./sign_xml_files/crosswalk.xml')  # 7. 횡단보도
except Exception as e:
    print(e)
    exit(1)


def u_turn_detect(image):
    # gray 이미지 변환
    gray_img = get_gray_img(image)
    u_turn = u_turn_cascade.detectMultiScale(gray_img, 1.06, 5)
    for (x, y, w, h) in u_turn:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print(np.matrix(u_turn))
    return image


def parking_detect(image):
    # gray 이미지 변환
    gray_img = get_gray_img(image)
    parking = parking_cascade.detectMultiScale(gray_img, 1.05, 5)
    for (x, y, w, h) in parking:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print(np.matrix(parking))
    return image


def get_gray_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    # 이미지 읽어오기
    cam = cv2.VideoCapture("./20170507_one_lap_normal_Moment_Parking.jpg")
    get_ok, image = cam.read()
    # 이미지 중 표지판이 있는 곳 확인
    # 표지판 종류 인식 결과 확인
    cv2.imshow('test', parking_detect(image))
    if cv2.waitKey(0) & 0xff == 27:
        cam.release()
        cv2.destroyAllWindows()
