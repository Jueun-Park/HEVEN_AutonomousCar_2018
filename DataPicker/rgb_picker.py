import cv2
import csv
import time

# 함수
# 이벤트 발생한 좌표에서 rgb를 반환
# 인자: 프레임(넘파이어레이), 마우스 클릭 이벤트
# 반환: rgb
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(tuple(param[y][x]))
        writer.writerow(tuple(param[y][x]))

def bind(win_name, frame):
    cv2.setMouseCallback(win_name, draw_circle, frame)

# 영상을 불러온다
cap = cv2.VideoCapture('./blue.mp4')
ret, img = cap.read()
cv2.namedWindow('image')
bind('image', img)

# loop
now_time = time.localtime()
now_time = "%04d-%02d-%02d+%02d-%02d-%02d" \
           % (now_time.tm_year, now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec)
file = open("rgb+data+"+ now_time + ".csv", 'a')
writer = csv.writer(file, lineterminator='\n')

while True:
    if img is None:
        print("image is none")
    else:
    # 이벤트 발생한 좌표에서 rgb를 파일 아웃풋으로 저장한다
    # 인자: 프레임(넘파이어레이), 마우스 클릭 이벤트


    # 한 프레임씩 스페이스바로 넘긴다
        cv2.imshow('image', img)
        if cv2.waitKey(0) & 0xFF == ord(' '):
            ret, img = cap.read()

        if cv2.waitKey(0) & 0xFF == ord('q'): break

file.close()