# !작년 학습 모델로 만들었고 현재 이 프로그램은 잘 안 됨!
# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)


import cv2
import numpy as np
import time
from shape_detection import shape_detect

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
    try:
        for (x, y, w, h) in square_array:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image
    except Exception:
        return image


def is_in_this_mission(ndarray):
    try:
        if np.sum(ndarray) >= 1:
            return True
        else:
            return False
    except Exception as e:
        return False


# mode dictionary (mission_name: mode_no)
modes = {'DEFAULT': 0,
         'PARKING': 1,
         'STATIC_OBS': 2,
         'MOVING_OBS': 3,
         'S_CURVE': 4,
         'NARROW': 5,
         'U_TURN': 6,
         'CROSS_WALK': 7, }

# mission name dictionary (mode_no: mission_name)
mission_name_dic = {0: 'DEFAULT',
                    1: 'PARKING',
                    2: 'STATIC_OBS',
                    3: 'MOVING_OBS',
                    4: 'S_CURVE',
                    5: 'NARROW',
                    6: 'U_TURN',
                    7: 'CROSS_WALK', }

# 표지판 감지기 인스턴스들 생성 (mode_no: sign_detector)
detector_dic = {0: None,
                1: SignDetector(parking_cascade, 1.3, 5),
                2: None,
                3: SignDetector(moving_cascade, 1.03, 5),
                4: None,
                5: None,
                6: SignDetector(u_turn_cascade, 1.06, 5),
                7: None, }

# 표지판 감지한 위치 담는 딕셔너리 (mode_no: location_data_array)
signboard_location_data = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None}


def process_one_frame_sign(frame, is_in_mission):
    if is_in_mission:
        pass

    t1 = time.time()  # 프레임 시작 시간 측정

    # 그래픽카드로 돌려보자? 쿠다 깔려 있어야 하는 듯?
    # frame = cv2.UMat(frame)

    for mode_no in detector_dic:  # 감지기들이 표지판 인식
        if detector_dic[mode_no]:
            # 각 미션에 해당하는 표지판 감지기가 위치 어레이를 반환하면 data dictionary 에 담는다.
            signboard_location_data[mode_no] = detector_dic[mode_no].detect(frame)
        else:
            continue

    now_mode_no = 0
    for mode_no in signboard_location_data:  # 데이터 가지고
        # 어떤 표지판 인식했는지 확인
        if is_in_this_mission(signboard_location_data[mode_no]):
            now_mode_no = mode_no
        frame = draw_square_on_image(frame, signboard_location_data[mode_no])  # 인식한 자리에 네모 표시

    # <여기서 수정할 내용>
    # 1. 정확도 문제 때문에, 연속으로 세 프레임 이상 감지해야 실제로 미션에 진입했다고 표시해 주는 기능 필요
    # - 딕셔너리에 기록하고 인풋 받고 다시 리턴하고를 반복하자.?
    # 2. 이미 지나친 미션에 대해서는 디텍터 인스턴스를 삭제해 버려서 다시 검사 안 하도록 하면 연산 속도 늘릴 수 있을 듯
    # 3. 미션에 진입한 이후에는 is_in_mission 리턴값 참조하여 그 동안에는 메서드 실행 안 하도록 해 줘야 한다. (병렬 처리?)

    if now_mode_no > 0:  # 감지 여부 출력
        print(now_mode_no, mission_name_dic[now_mode_no], ": DETECT!")
        is_in_mission = True
    else:
        print("defalt mode", now_mode_no)

    cv2.imshow('test', frame)  # 인식 된 곳에 네모 그려둔 것 표시
    t2 = time.time()  # 프레임 종료 시간 측정
    print("time per frame:", t2 - t1)
    return now_mode_no, is_in_mission


# TEST CODE
if __name__ == "__main__":

    # 웹캠 읽어오기
    cam = cv2.VideoCapture(0)
    time.sleep(2)

    is_in_mission = False
    # 영상 처리
    while (True):
        frame_okay, frame = cam.read()  # 한 프레임을 가져오자.
        # 이미지 중 표지판이 있는 곳 확인
        img_list = shape_detect(frame)
        for img in img_list:
            process_one_frame_sign(img, is_in_mission)

        if cv2.waitKey(1) & 0xff == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break
