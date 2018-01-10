# 카메라 통신 및 표지판 인식
# input: sign_cam
# output: 표지판 종류 (to car_control)

import cv2

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
    # 변수명 = cv2.CascadeClassifier('./xml_directory/파일 이름')
    # xml_dic 에 들어 있는 표지판 개수만큼 실행
    exec(i + ' = ' + 'cv2.CascadeClassifier(\'' + xml_directory + xml_dic[i] + '\')')

    while True:
        ret, frame = sign_cam.read()
        cv2.imshow('image', frame)

        s, img = sign_cam.read()  # for test
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로
        crosswalk = crosswalk_cascade.detectMultiScale(gray, 1.1, 20)  # 파일 오픈 확인...

        if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
            break

    sign_cam.release()
    cv2.destroyAllWindows()
