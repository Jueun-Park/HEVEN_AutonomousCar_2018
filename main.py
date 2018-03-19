# 2018 국제대학생창작자동차대회 자율주행차 부문 성균관대학교 HEVEN 팀
# interpreted by python 3.6
# 하위 프로그램


#module
import lidarstart
import threading
from multiprocessing import Process


# cpu 병렬 처리

def main():

    #lane_detection_thread=threading.Thread(target=lane_cam_noclass) 미완성
    lidarstart_thread=Process(target=lidarstart.lidarstart())

    #sign_cam=threading.Thread(target=signmain.main())
    #sign_cam_cros=threading.Thread(target=sign_cam2.crosswalk_detect())
    #sign_cam_thread = threading.Thread(target=sign_cam.main())
    #communicationstart_thread=threading.Thread(target=communicationstart.communicationstart())
    lidarstart_thread.start()
    #sign_cam.start()

    #sign_cam_cros.start()





    #communicationstart_thread.start()







    '''
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
    '''


# 센서 값 받기
# 라이다
# 차선 비전
# 표지판 비전

# 플래닝
# 제어

# 통신

if __name__ == "__main__":
    main()


