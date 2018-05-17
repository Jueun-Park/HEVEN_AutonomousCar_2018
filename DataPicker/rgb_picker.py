# 2018-05-17
# 영상에서 클릭한 좌표의 rgb값을 csv 파일에 기록하는 프로그램
# 개발자: 김홍빈 이아영 박주은

import cv2
import csv
import time
import numpy as np


# 함수
# 클릭 이벤트 발생한 좌표에서 rgb를 반환
# 인자: 프레임(넘파이어레이), 마우스 클릭 이벤트
# 반환: rgb
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(param[y][x])
        writer.writerow(tuple(param[y][x]))  # csv 파일에 작성


def bind(win_name, frame):
    cv2.setMouseCallback(win_name, draw_circle, frame)


# 영상을 불러온다
cap = cv2.VideoCapture('./blue_2.mp4')  # video name
ret, img = cap.read()
cv2.namedWindow('image')
bind('image', img)
# 클릭 이벤트 발생한 좌표에서 rgb를 반환하여 기록


# open csv file for writing
now_time = time.localtime()  # 현재 시간을 파일명에 추가한다
now_time = "%04d-%02d-%02d+%02d-%02d-%02d" \
           % (now_time.tm_year, now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec)
file = open("rgb+data+" + now_time + ".csv", 'a')
writer = csv.writer(file, lineterminator='\n')  # rgb 단위로 줄바꿈

# loop
while True:
    if img is None:
        print("image is none")
        break
    else:
        # 한 프레임씩 스페이스바로 넘긴다
        cv2.imshow('image', img)
        if cv2.waitKey(0) & 0xFF == ord(' '):
            ret, img = cap.read()
            bind('image', img)

        # press 'q' for exit
        if cv2.waitKey(0) & 0xFF == ord('q'): break

file.close()
