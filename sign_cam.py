# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)

import cv2

# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
#          'MOVING_OBS': 3,'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

parking_cascade = cv2.CascadeClassifier('./sign_xml_files/parkingdetect.xml')  # 1. 자동 주차
static_cascade = cv2.CascadeClassifier('./sign_xml_files/static_0514.xml')  # 2. 정적 장애물
moving_cascade = cv2.CascadeClassifier('./sign_xml_files/moving_0510.xml')  # 3. 동적 장애물
s_curve_cascade = cv2.CascadeClassifier('./sign_xml_files/scurve_0517.xml')  # 4. S자 주행
narrow_cascade = cv2.CascadeClassifier('./sign_xml_files/narrowno.xml')  # 5. 협로 주행
u_turn_cascade = cv2.CascadeClassifier('./sign_xml_files/uturndetect.xml')  # 6. 유턴
crosswalk_cascade = cv2.CascadeClassifier('./sign_xml_files/crosswalk.xml')  # 7. 횡단보도


# 웹캠인 경우: capture = cv2.VideoCapture(0), 0은 주소값
# 비디오인 경우: capture = cv2.VideoCapture('video_test.avi'), 파일명
def show_video(capture):
    while True:
        # ret: frame capture 결과(boolean)
        # frame: Capture 한 frame
        ret, frame = capture.read()
        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
            break

    capture.release()
    cv2.destroyAllWindows()


class SignDetection:
    def __init__(self, frame, cascade, mission_no):
        self.stop = 0
        self.is_in_mission = False
        self.mission_no = mission_no
        self.frame = frame
        self.cascade = cascade

        self.parking_stop = 0
        self.static_stop = 0
        self.moving_stop = 0
        self.s_curve_stop = 0
        self.narrow_stop = 0
        self.u_turn_stop = 0
        self.crosswalk_stop = 0

        self.detect_crosswalk = 0
        self.detect_narrow = 0
        self.detect_moving = 0
        self.detect_static = 0
        self.detect_s_curve = 0
        self.detect_u_turn = 0
        self.detect_parking = 0

    def cam_open(self):
        pass


    def detect(self):
        if self.stop == 3 or self.is_in_mission:
            pass
        else:
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            print("Stop is ", self.stop)

            # 흑백 이미지에서 횡단보도_표지판_위치_리스트 추출
            sign = self.cascade.detectMultiScale(image=gray, scaleFactor=1.1, minNeighbers=20)

            for (x, y, w, h) in sign:  # 횡단보도_표지판_위치_리스트 위치 화면에 사각형으로 표시
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if len(sign) >= 1:  # 하나라도 감지된 상태이면
                self.stop += 1
                print("DETECTED!!! ", self.mission_no)

                if self.stop == 3:
                    self.is_in_mission = True

    def test_main(self):
        pass


def main():
    capture = cv2.VideoCapture(0)

    # method for test
    # show_video(capture)

    ret, frame = capture.read()

    # make instance
    u_turn_detection = SignDetection(frame, u_turn_cascade, 1)

    while True:
        u_turn_detection.detect()

        cv2.imshow("Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
