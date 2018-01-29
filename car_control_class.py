import time
import math


class Steering:

    front_dis = 0.5  ## 임의로 거리 지정 (실험값 필요)
    car_front = 0.28
    car_dis = 0.78  ## front_dis + car_front
    velocity = 1.5

    x = 2.0 ## 임의의 값 설정
    y = 0.6 ## 임의의 값 설정

    def __init__(self, linear, cross_track_error, stop_line):
        self.linear = linear
        self.cross_track_error = cross_track_error/100
        self.stop_line = stop_line
        self.t1 = 0
        self.t2 = 0
        self.gear = 0
        self.steer = 0
        self.steer_past = 0
        self.speed = 54

    def steer_s(self):
        tan_value = self.linear * (-1)
        theta_1 = math.degrees(math.atan(tan_value))

        k = 1
        if -15 < theta_1 < 15 and abs(self.cross_track_error) < 0.27:
                k = 0.5

        theta_2 = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))
        steer_now = (theta_1 + theta_2)
        adjust = 0.3
        steer_final = (adjust * self.steer_past) + ((1 - adjust) * steer_now)

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

    def cross_walk(self):
        if abs(self.stop_line) / 100 < 1:  ## 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if self.t1 == 0:
                self.t1 = time.time()
            self.t2 = time.time()

            if (self.t2 - self.t1) < 3.0:  ## 3초간 정지, 실험을 통해 보정 필요
                self.speed = 0
