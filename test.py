# 하위 프로그램
import lidarstart



modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
         'MOVING_OBS': 3, 'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

from pycuda.compiler import SourceModule
#module
import threading

# cpu 병렬 처리



def main():



    # lane_detection_thread=threading.Thread(target=lane_cam_noclass) 미완성
    #sign_cam_thread = threading.Thread(target=sign_cam.main())



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
    lidarstart.lidarstart()


