
# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2  # opencv 사용
import numpy as np


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def mark_img(img, blue_threshold=160, green_threshold=160, red_threshold=160):  # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (img[:, :, 0] < bgr_threshold[0]) \
                 | (img[:, :, 1] < bgr_threshold[1]) \
                 | (img[:, :, 2] < bgr_threshold[2])
    mark[thresholds] = [0, 0, 0]
    return mark



#CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/Parking Detection.mp4'
CAM_ID = 'C:/Users/jglee/Desktop/VIDEOS/0507_one_lap_normal.mp4'
#CAM_ID = 1

cam = cv2.VideoCapture(CAM_ID)  # 카메라 생성
if cam.isOpened() == False:  # 카메라 생성 확인
    print('Can\'t open the CAM')


# 카메라 이미지 해상도 얻기
cam.set(cv2.CAP_PROP_FRAME_WIDTH,480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,270)

width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('size = ', width, height)


while (True):
    ret, frame = cam.read()
    mark = np.copy(frame)  # roi_img 복사
    mark = mark_img(frame)  # 흰색 차선 찾기

    # 흰색 차선 검출한 부분을 원본 image에 overlap 하기
    color_thresholds = (mark[:, :, 0] == 0) & (mark[:, :, 1] == 0) & (mark[:, :, 2] > 200)
    frame[color_thresholds] = [0, 0, 255]

    cv2.imshow('roi_white', mark)  # 흰색 차선 추출 결과 출력
    cv2.imshow('result', frame)  # 이미지 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)

