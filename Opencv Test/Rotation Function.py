import cv2


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = null
    return dst




CAM_ID = 0

cam = cv2.VideoCapture(CAM_ID)  # 카메라 생성
if cam.isOpened() == False:  # 카메라 생성 확인
    print
    ('Can\'t open the CAM(%d)') % (CAM_ID)
    exit()

# 카메라 이미지 해상도 얻기
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print
('size = [%f, %f]\n') % (width, height)

# 윈도우 생성 및 사이즈 변경
cv2.namedWindow('CAM_OriginalWindow')
cv2.resizeWindow('CAM_OriginalWindow', 1280, 720)

########### 추가 ########################
# 회전 윈도우 생성
cv2.namedWindow('CAM_RotateWindow')
########################################

while (True):
    # 카메라에서 이미지 얻기
    ret, frame = cam.read()

    ########### 추가 ########################
    # 이미지를 회전시켜서 img로 돌려받음
    img = Rotate(frame, 270)  # 90 or 180 or 270
    ########################################

    # 얻어온 이미지 윈도우에 표시
    cv2.imshow('CAM_OriginalWindow', frame)

    ########### 추가 ########################
    # 회전된 이미지 표시
    cv2.imshow('CAM_RotateWindow', img)
    #########################################

    # 10ms 동안 키입력 대기
    if cv2.waitKey(10) >= 0:
        break;

# 윈도우 종료
cam.release()
cv2.destroyWindow('CAM_OriginalWindow')

########### 추가 ########################
cv2.destroyWindow('CAM_RotateWindow')
########################################
