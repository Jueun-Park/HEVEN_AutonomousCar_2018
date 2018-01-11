# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)

import cv2
import numpy as np

# 지난 대회 사용 안 함
narrow_cascade = cv2.CascadeClassifier('./sign_xml_files/narrowno.xml')
static_cascade = cv2.CascadeClassifier('./sign_xml_files/static_0514.xml')
s_curve_cascade = cv2.CascadeClassifier('./sign_xml_files/scurve_0517.xml')
parking_cascade = cv2.CascadeClassifier('./sign_xml_files/parkingdetect.xml')

# 지난 대회에 사용함
u_turn_cascade = cv2.CascadeClassifier('./sign_xml_files/uturndetect.xml')  # 1. 유턴
crosswalk_cascade = cv2.CascadeClassifier('./sign_xml_files/crosswalk.xml')  # 2. 횡단보도
moving_cascade = cv2.CascadeClassifier('./sign_xml_files/moving_0510.xml')  # 3. 동적 장애물

cascade_dic = {1: u_turn_cascade, 2: crosswalk_cascade, 3: moving_cascade}
stop_dic = {1: 0, 2: 0, 3: 0}


# 웹캠인 경우: capture = cv2.VideoCapture(0), 0은 주소값
# 비디오인 경우: capture = cv2.VideoCapture('vtest.avi'), 파일명
def show_video(capture):
    while True:
        # ret : frame capture 결과(boolean)
        # frame : Capture 한 frame
        ret, frame = capture.read()
        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
            break

    capture.release()
    cv2.destroyAllWindows()


# int stop, int stop_key, bool is_in_mission, int mission
# ret, frame = capture.read()
def detect_sign(stop, stop_key, is_in_mission, mission, frame, cascade):
    if stop == 3 or is_in_mission:  # 3번 멈췄거나, 미션을 수행 중이라면
        pass  # 함수를 실행하지 않는다
    else:  # 함수 실행
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로
        print(stop_key, "Stop is ", stop)

        # 흑백 이미지에서 표지판 추출
        sign = cascade.detectMultiScale(gray, 1.1, 20)

        for (x, y, w, h) in sign:  # 표지판 위치 화면에 사각형으로 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        rec1 = np.matrix(sign)  # 표지판을 감지한 위치를 저장

        if np.sum(rec1) >= 1:  # 하나라도 감지된 상태이면
            stop += 1

            if stop == 3:  # 3번째 감지할 때
                mission = 3
                is_in_mission = True  # 미션을 수행 중이다

            print("DETECTED!!! ", stop_key)

    return stop, stop_key, is_in_mission, mission, frame


def main():
    sign_cam = cv2.VideoCapture(0)  # 표지판을 볼 카메라에 연결할 것
    # show_video(sign_cam)

    stop = 0
    is_in_mission = False
    mission = 0

    while True:
        # ret: frame capture 결과(boolean)
        # frame: Capture 한 frame
        ret, frame = sign_cam.read()
        cv2.imshow('image', frame)

        for i in cascade_dic:
            stop_dic[i], key, is_in_mission, mission, frame = detect_sign(stop_dic[i], i, is_in_mission, mission, frame, cascade_dic[i])
            if is_in_mission:
                print(key, " is in mission")
                # 미션 수행... 근데 뭔가 이상한데? 리턴값 어케주냐 게터로주나 하... 헐그럼 클래스로짜야함?
                is_in_mission = False

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
            break

    sign_cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
