import time
import math


class Global:
    t1 = 0
    t2 = 0
    gear = 0
    steer = 0
    steer_past = 0
    speed = 54


class Steering(Global):

    front_dis = 0.5  ## 임의로 거리 지정 (실험값 필요)
    car_front = 0.28
    car_dis = 0.78  ## front_dis + car_front
    velocity = 1.5

    def __init__(self, linear, cross_track_error, stop_line, obs_pos):
        self.linear = linear
        self.cross_track_error = cross_track_error/100
        self.stop_line = stop_line
        self.obs_pos = obs_pos

    def steer_s(self):
        tan_value = self.linear * (-1)
        theta_1 = math.degrees(math.atan(tan_value))

        k = 1
        if -15 < theta_1 < 15:
            if abs(self.cross_track_error) < 0.27:
                k = 0.5

        theta_2 = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))
        steer_now = (theta_1 + theta_2)
        adjust = 0.3
        steer_final = (adjust * Global.steer_past) + ((1 - adjust) * steer_now)

        Global.steer = steer_final * 71

        Global.steer_past = steer_final

        if Global.steer > 1970:
            Global.steer = 1970
            Global.steer_past = 27.746
        elif Global.steer < -1970:
            Global.steer = -1970
            Global.steer_past = -27.746

    def cross_walk(self):
        if abs(self.stop_line) / 100 < 1:  ## 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if Global.t1 == 0:
                Global.t1 = time.time()
            Global.t2 = time.time()

            if (Global.t2 - Global.t1) < 3.0:  ## 3초간 정지, 실험을 통해 보정 필요
                Global.speed = 0

    def moving_obs(self):
        if self.obs_pos[0] == 0 and self.obs_pos[1] == 0:
            Global.speed = 54
        else:
            obs_x = round(self.obs_pos[0] * math.cos(self.obs_pos[1]), 2)
            obs_y = round(self.obs_pos[0] * math.sin(self.obs_pos[1]), 2)

            if abs(obs_y) / 100 < 2:
                if abs(obs_x) / 100 < 1.5:
                    Global.speed = 0