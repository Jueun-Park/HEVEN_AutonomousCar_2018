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

global gear, a, speed_Obs, steer_past, ENC1
a = [0, 0]
ENC1 = 0
gear = 0
steer = 0
steer_past = 0
obst_speed = 54
main_speed = 54


################Function#######################
def steering(Mission, ch, curvature, linear, cross_track_error_1, cross_track_error_2, stop_line,
             obs_dis):  # , speed_Default, speed_Obs):

    global steer, gear, speed_Obs, speed_Default
    global steer_past
    check = Mission * ch

    ################### U-TURN ##################################
    if check == 5:
        speed_Obs = 40
        gear = 0
        ch = 2
        Mission = 9
        return ch, steer, speed_Obs, Mission, gear

    ################### S-CURVE #################################
    elif check == 0:
        speed_Obs = 30
        front_dis = 0.10  ## 임의로 거리 지정 (실험값 필요)
        car_front = 0.28
        car_dis = front_dis + car_front
        all_dis_1 = round(pow(car_dis, 2), 2) + round(pow(cross_track_error_2 / 100, 2), 2)
        all_dis_2 = round(math.sqrt(all_dis_1), 2)
        velocity = 0.83
        tan_value = linear / (front_dis + car_front)
        theta_1 = math.degrees(math.atan(tan_value))
        speed_Default = main_speed
        k = 1
        if -15 < theta_1 < 15:
            if abs(cross_track_error_1) / 100 < 0.16:
                k = 0.5
        theta_2 = math.degrees(math.atan((k * cross_track_error_1) / velocity))
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
        return ch, steer, speed_Obs, Mission, gear


    ################  JU - CHA ##################################

    elif check == 7:


        steer = 0
        gear = 0
        check = 10
        return ch, steer, speed_Obs, Mission, gear

    ################ CROSS - WALK  ###############################

    elif check == 1:  ## 정확한 값을 몰라 임의의 값 설정, 후에 필히 조정 바람
        steer = 0
        gear = 0
        if stop_line < 1:  ## 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if a[0] == 0:
                a[0] = time.time()
            a[1] = time.time()

            if (a[1]-a[0]) > 3.0: ## 3초간 정지, 실험값 보정 필요
                a[0] == 0
                a[1] == 0
                speed_Obs = main_speed
                ch = 2  ## 미션에서 벗어나도록 명령, 임의값 설정, 필히 조정 바람
            else:
                speed_Obs = 0
        return ch, steer, speed_Obs, Mission, gear

    ################ Moving - Obs  ###############################

    elif check == 7:
        gear = 0
        steer = 0
        speed_Obs = obst_speed
        if obs_dis < 1: ## 일정거리 앞에 장애물 감지시 정지, 임의의 값, 필히 조정 바람
            speed_Obs = 0
        elif obs_dis > 2: ## 장매물이 완전히 벗어났을 때 움지기임, 임의의 값, 필히 조정 바람
            speed_Obs = main_speed
            ch = 2  ## 미션에서 벗어나도록 명령, 임의값 설정, 필히 조정 바람
        return ch, steer, speed_Obs, Mission, gear

    ################ default ######################################

    else:
        gear = 0
        front_dis = 0.5  ## 임의로 거리 지정 (실험값 필요)
        car_front = 0.28
        car_dis = front_dis + car_front
        all_dis_1 = round(pow(car_dis, 2), 2) + round(pow(cross_track_error_2 / 100, 2), 2)
        all_dis_2 = round(math.sqrt(all_dis_1), 2)
        velocity = 1.5
        tan_value = (linear * (-1)) / (front_dis + car_front)
        theta_1 = math.degrees(math.atan(tan_value))
        speed_Default = main_speed
        k = 1
        if -15 < theta_1 < 15:
            if abs(cross_track_error_1) / 100 < 0.27:
                k = 0.5
        theta_2 = math.degrees(math.atan((k * cross_track_error_1) / velocity))
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
        print(steer)
        return ch, steer, speed_Default, Mission, gear