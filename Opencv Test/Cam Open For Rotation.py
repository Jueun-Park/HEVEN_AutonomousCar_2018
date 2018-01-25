# -*- Encoding:UTF-8 -*- #
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

def camopen(CAM_ID):
    cam = cv2.VideoCapture(CAM_ID)  # 카메라 생성
    if cam.isOpened() == False:  # 카메라 생성 확인
        print('Can\'t open the CAM')
        exit()

    # 카메라 이미지 해상도 얻기
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('size = ', width, height)



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

        # Q 누르기 전까지 작동.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

CAM_ID = 0
camopen(CAM_ID)
