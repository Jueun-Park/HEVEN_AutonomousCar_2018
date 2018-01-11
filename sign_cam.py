# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)

import cv2
import numpy as np


def main():
    sign_cam = cv2.VideoCapture(0)  # 표지판을 볼 카메라에 연결할 것

    # 머신러닝 된 xml 파일 목록 (표지판:xml 파일 이름)
    xml_dic = {'crosswalk_cascade': 'crosswalk.xml',
               'narrow_cascade': 'narrowno.xml',
               'moving_cascade': 'moving_0510.xml',
               'static_cascade': 'static_0514.xml',
               's_curve_cascade': 'scurve_0514.xml',
               'u_turn_cascade': 'uturndetect.xml',
               'parking_cascade': 'parkingdetect.xml'}
    xml_directory = './sign_xml_files/'  # xml 파일 디렉토리

    for i in xml_dic.keys():
        # 문자열로 된 statement 실행, xml 파일 오픈
        # xml_dic 에 들어 있는 표지판 개수만큼 실행
        print(i)  # debugging
        # 변수명 = cv2.CascadeClassifier('./xml_directory/파일 이름')
        exec(i + ' = ' + 'cv2.CascadeClassifier(\'' + xml_directory + xml_dic[i] + '\')')

    for i in xml_dic.keys():
        stop = {i: 0}

    while True:
        ret, frame = sign_cam.read()
        cv2.imshow('image', frame)

        s, image = sign_cam.read()
        del s
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로

        for i in xml_dic.keys():
            detect_sign(image, gray, i, stop[i], i)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
            break

    sign_cam.release()
    cv2.destroyAllWindows()


def detect_sign(image, gray, cascade, stop, sign_no):
    aaaaaaaa = cascade.detectMultiScale(gray, 1.1, 20)
    for (x, y, w, h) in aaaaaaaa:  # 표지판 위치 화면에 사각형으로 표시
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    rec1 = np.matrix(aaaaaaaa)  # 표지판을 감지한 위치를 저장

    if np.sum(rec1) >= 1:  # 하나라도 감지된 상태이면
        detect_crosswalk = 1  # 감지된 상태이다. 표시
        stop[sign_no] += 1  # global

        if stop[sign_no] == 3:  # 3번째 감지할 때
            Mission = 3  # global
            in_Mission = True  # global, 미션을 수행 중이다

        print("CrossWalk!!! ", detect_crosswalk)


if __name__ == '__main__':
    main()
