# 차량 제어 (기본, 미션별)
# input: 1. 경로가 표시된 numpy array (from path_planner)
#        2. 인식한 표지판 정보 (from sign_cam)
#        3. 통신 패킷 정보, 형식 미정 (from communication)
# output: 통신 패킷 만드는 데 필요한 정보 (to communication)

import time
import math


class Steering:

    front_dis = 0.5  # 임의로 거리 지정 (실험값 필요)
    car_front = 0.28
    car_dis = 0.78  # front_dis + car_front
    velocity = 1.5

    def __init__(self, mission, linear, cross_track_error, stop_line, obs):
        self.mission = mission
        self.linear = linear

        self.cross_track_error = cross_track_error/100
        self.stop_line = stop_line

        self.t1 = 0
        self.t2 = 0

        self.gear = 0
        self.steer = 0
        self.steer_past = 0
        self.speed = 54

        self.dis = obs[0]
        self.y = obs[1]

        self.theta_1 = 0
        self.theta_2 = 0
        self.theta_3 = 0
        self.tan_value = 0
        self.adjust = 0

    def steer_s(self):
        if self.linear is None and self.cross_track_error is None:
            self.tan_value = (abs(self.dis) / self.y)
            self.theta_3 = math.degrees(math.atan(self.tan_value))

            if self.dis < 0:
                self.theta_3 = self.theta_3 * (-1)
            else:
                self.theta_3 = self.theta_3

            self.theta_1 = 0
            self.theta_2 = 0
            self.adjust = 0.1
        else:
            self.tan_value = self.linear * (-1)
            self.theta_1 = math.degrees(math.atan(self.tan_value))

            k = 1
            if -15 < self.theta_1 < 15 and abs(self.cross_track_error) < 0.27:
                k = 0.5

            self.theta_2 = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))

            self.theta_3 = 0
            self.adjust = 0.3

        steer_now = (self.theta_1 + self.theta_2 + self.theta_3)
        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * steer_now)

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

    def cross_walk(self):
        if abs(self.stop_line) / 100 < 1:  # 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if self.t1 == 0:
                self.t1 = time.time()
            self.t2 = time.time()

            if (self.t2 - self.t1) < 3.0:  # 3초간 정지, 실험을 통해 보정 필요
                self.speed = 0
            else:
                self.speed = 54

