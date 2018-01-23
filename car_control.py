# 차량 제어 (기본, 미션별)
# input: 1. 경로가 표시된 numpy array (from path_planner)
#        2. 인식한 표지판 정보 (from sign_cam)
#        3. 통신 패킷 정보, 형식 미정 (from communication)
# output: 통신 패킷 만드는 데 필요한 정보 (to communication)

import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt

global gear, a, steer_past, ENC1
a = [0, 0]
ENC1 = 0
gear = 0
steer = 0
steer_past = 0
main_speed = 54

#
################Function######################
def steering (linear, cross_track_error, stop_line, obs_pos)
    global steer, gear, speed_Obs, speed_Default
    global steer_past
    ## obs_pos : (r,theta)값으로 받아오기

    ################### S-CURVE #################################
    if check == 0:
        speed_Obs = 30
        front_dis = 0.10  ## 임의로 거리 지정 (실험값 필요)
        car_front = 0.28
        car_dis = front_dis + car_front
        velocity = 0.83
        tan_value = linear / car_dis
        theta_1 = math.degrees(math.atan(tan_value))
        speed_Default = main_speed
        k = 1
        if -15 < theta_1 < 15:
            if abs(cross_track_error) / 100 < 0.16:
                k = 0.5
        theta_2 = math.degrees(math.atan((k * cross_track_error) / velocity))
        steer_now = theta_1 + theta_2
        adjust = 0.3
        steer_final = (adjust * steer_past) + ((1 - adjust) * steer_now)
        steer = steer_final * 71
        steer_past = steer_final
        if steer > 1970:
            steer = 1970
        elif steer < -1970:
            steer = -1970
        ##        print(cross_track_error)
        ##        print(steer_final)
        ##        print(steer)
        return steer, speed_Obs, gear

    ################ default ######################################

    else:
        obs_x = (obs_pos[0]*math.cos(obs_pos[1]))
        obs_y = (obs_pos[0]*math.sin(obs_pos[1]))
        ##########################################################
        if obs_y < 2:
            if obs_x < 1.5:
                speed_Default = 0
        ##########################################################
        if stop_line/100 < 1:  ## 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if a[0] == 0:
                a[0] = time.time()
            a[1] = time.time()

            if (a[1]-a[0]) < 3.0: ## 3초간 정지, 실험값 보정 필요
                speed_Default = 0
        ##########################################################
        gear = 0
        front_dis = 0.5  ## 임의로 거리 지정 (실험값 필요)
        car_front = 0.28
        car_dis = front_dis + car_front
        velocity = 1.5
        tan_value = (linear * (-1)) / car_dis
        theta_1 = math.degrees(math.atan(tan_value))
        speed_Default = main_speed
        k = 1
        if -15 < theta_1 < 15:
            if abs(cross_track_error) / 100 < 0.27:
                k = 0.5
        theta_2 = math.degrees(math.atan((k * cross_track_error) / velocity))
        steer_now = theta_1 + theta_2
        adjust = 0.3
        steer_final = (adjust * steer_past) + ((1 - adjust) * steer_now)
        steer = steer_final * 71
        steer_past = steer_final
        if steer > 1970:
            steer = 1970
        elif steer < -1970:
            steer = -1970
        ##        print(cross_track_error)
        ##        print(steer_final)
        ##        print(steer)
        return steer, speed_Default, gear