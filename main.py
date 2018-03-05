# 2018 국제대학생창작자동차대회 자율주행차 부문 성균관대학교 HEVEN 팀
# interpreted by python 3.6

# 하위 프로그램
import lidar
import sign_cam
import lane_cam_noclass
import path_planner
import car_control
import communication

# module
import numpy
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from multiprocessing import pool, process
import threading
import time
from pycuda.compiler import SourceModule
# cpu 병렬 처리
"""
def getFunction(name, contents): # name은 함수의 이름, contents는 함수의 본체여야 합니다.
    mod = SourceModule(contents) # mod에 contents를 불러옵니다.
    return mod.get_function(name) # mod에서 이름이 name인 함수를 반환합니다.
"""

def main():
    lane_detection_thread=threading.Thread(target=lane_cam_noclass.lane_Detection())
    sign_cam_thread=threading.Thread(target=sign_cam.main())
    read_lidar_thread=threading.Thread(target=lidar.initiate())






    # 센서 값 받기

    # 라이다
    # 차선 비전
    # 표지판 비전

    # 플래닝
    # 제어

    # 통신

    pass

if __name__ == "__main__" :

    start_time=0
    end_time=0

    while True :
        start_time2 = time.time()
        print(start_time,end_time)
        end_time = time.time()
        # openCam()
        main()



    cv2.destroyAllWindows()
    cv2.waitKey(0)


