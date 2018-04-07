# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)

# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
#          'MOVING_OBS': 3,'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

import cv2
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


class SignDetector:
    def __init__(self, cascade, scale_factor, min_neighbors):
        self.cascade = cascade
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image):
        gray_image = get_gray_img(image)
        detected_array = self.cascade.detectMultiScale(gray_image, self.scale_factor, self.min_neighbors)
        return detected_array


def get_gray_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_square_on_image(image, square_array):
    for (x, y, w, h) in square_array:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image


if __name__ == "__main__":
    # 이미지 읽어오기
    cam = cv2.VideoCapture("./20170507_one_lap_normal_Moment_Parking.jpg")
    t1 = time.time()
    get_ok, image = cam.read()

    # 그래픽카드로 돌려보자? 쿠다 깔려 있어야 하는 듯?
    # image = cv2.UMat(image)

    # 표지판 감지기 인스턴스들 생성
    parking_detector = SignDetector(parking_cascade, 1.05, 5)
    u_turn_detector = SignDetector(u_turn_cascade, 1.06, 5)

    # 이미지 중 표지판이 있는 곳 확인

    # 표지판을 인식해라 감지기들이여.
    parking_detected_array = parking_detector.detect(image)
    u_turn_detected_array = u_turn_detector.detect(image)

    detected_image = draw_square_on_image(image, parking_detected_array)
    detected_image = draw_square_on_image(detected_image, u_turn_detected_array)
    t2 = time.time()

    # 인식 된 곳에 네모 그려둔 것 표시
    cv2.imshow('test', detected_image)

    print(t2 - t1)

    if cv2.waitKey(0) & 0xff == 27:
        cam.release()
        cv2.destroyAllWindows()
