# 2018 국제대학생창작자동차대회 자율주행차 부문 성균관대학교 HEVEN 팀
# interpreted by python 3.6

# 하위 프로그램
import lidar
import sign_cam
import lane_cam
import path_planner
import car_control
import communication
import lane_cam_noclass
# module
import numpy
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
from multiprocessing import pool, process
import threading



from pycuda.compiler import SourceModule

# cpu 병렬 처리

def getFunction(name, contents): # name은 함수의 이름, contents는 함수의 본체여야 합니다.
    mod = SourceModule(contents) # mod에 contents를 불러옵니다.
    return mod.get_function(name) # mod에서 이름이 name인 함수를 반환합니다.


def main():

    lane_detection_thread=threading.Thread(target=lane_cam_noclass)
    sign_cam_thread=threading.Thread(target=sign_cam.main())
    read_lidar_thread1=threading.Thread(target=lidar.current_lidar.set_ROI(80,300))#클래스로 되어있어 스레딩 처리 따로 되어있어야할듯
    read_lidar_thread2=threading.Thread(target=lidar.current_lidar.initiate())
    read_lidar_thread3=threading.Thread(target=lidar.current_lidar.set_mode(1))
    read_lidar_thread4=threading.Thread(target=lidar.current_lidar.plot_data())#물어보고 진행







    Lidar_thread = threading.Thread(target=read_Lidar())
    Matrix_thread = threading.Thread(target=matrix_Comb())
    AS_thread = threading.Thread(target=astar_Comb())
    # IMU_thread = threading.Thread(target = read_IMU())
    PFread_thread = threading.Thread(target=read_PF())
    # GPS_thread = threading.Thread(target = read_GPS())
    STEER_thread = threading.Thread(target=steer_Comb())
    PFwrite_thread = threading.Thread(target=write_PF())
    Show_thread = threading.Thread(target=show_Path())
    Obs_thread = threading.Thread(target=front_Detect())
    # U_thread = threading.Thread(target = uturn_detect())
>>>>>>> Stashed changes
    # 센서 값 받기
    # 라이다
    # 차선 비전
    # 표지판 비전

    # 플래닝
    # 제어

    # 통신

<<<<<<< Updated upstream
    pass
=======
    pass

if __name__ == "__main__":
    start_time = 0
    end_time = 0
    while True:
        start_time = time.time()
        print(start_time - end_time)
        end_time= time.time()
        # openCam()
        main()
        # astar_Comb()
        dim = (500, 500)
        # map_plot = cv2.resize(lane_Comb, dim, interpolation = cv2.INTER_AREA)
        # lidar_plot = cv2.resize(lidar_Comb, dim, interpolation = cv2.INTER_AREA)
        lane_plot = cv2.resize(matrix_Show, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('map', map_plot)
        # cv2.imshow('lidar', lidar_plot)
        cv2.imshow('Path', lane_plot)
        # cv2.imshow('cam', img)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)

