import cv2
import csv
import time

# 영상을 불러온다
cap = cv2.VideoCapture('C:/Users/LG/PycharmProjects/untitled6/sign_logging_12.avi')
ret, img = cap.read()

# loop
while(True):

    if img is None:
        print("image is none")
    else:
    # 이벤트 발생한 좌표에서 rgb를 파일 아웃풋으로 저장한다
    # 인자: 프레임(넘파이어레이), 마우스 클릭 이벤트


    # 한 프레임씩 스페이스바로 넘긴다
        if cv2.waitKey(0) & 0xFF == ord(' '):
            ret, img = cap.read()
        cv2.imshow('img', img)

# 함수
# 이벤트 발생한 좌표에서 rgb를 반환
# 인자: 프레임(넘파이어레이), 마우스 클릭 이벤트
# 반환: rgb

# 함수
# 파일 아웃풋으로 저장한다
# 인자: 기록할 값: rgb, 튜플
# 반환: csv
def write_rgb_to_csv(rgb):
    now_time = time.localtime()
    now_time = "%04d-%02d-%02d+%02d-%02d-%02d" \
               % (now_time.tm_year, now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec)
    file = open("rgb+data+"+ now_time + ".csv", 'a')
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(list(rgb))
    file.close()