# 엔코더 앞바퀴 왼쪽에 부착
# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
#           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

# platform에서 받는 데이터 처리(현재 주행 속도, 조향각, 엔코더 값)

import time
import math


class Control:

    velocity = 1.5
    car_front = 0.28  # 수정 바람 - 차량 정지 시간

    def __init__(self, mission_num, first, second):
        self.gear = 0
        self.speed = 0
        self.steer = 0
        self.brake = 0

        self.velocity = 0
        self.steer_past = 0

        self.default_y_dis = 1  # (임의의 값 / 1m)

        #######################################
        # communication.py 에서 데이터 받아오기#
        self.speed_platform = 0               #
        self.ENC1 = [0, 0]                    #
        #######################################

        self.mission_num = mission_num

        if self.mission_num == 0:
            self.cross_track_error = first/100
            self.linear = second[0]
            self.cul = second[1]

            self.__default__()

    def __default__(self):
        self.steer = 0
        self.speed = 54
        self.gear = 0
        self.brake = 0

        self.tan_value_1 = self.linear * (-1)
        self.theta_1 = math.atan(self.tan_value_1)

        self.son = 0.4925*(1-math.cos(self.theta_1)) + self.cross_track_error
        self.mother = 1.04 + self.default_y_dis + 0.4925*(math.sin(self.theta_1))

        self.tan_value_2 = abs(self.son / self.mother)
        self.theta_line = math.atan(self.tan_value_2)

        self.adjust = 0.1

        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * self.theta_line) * 1.387

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

        return self.steer, self.speed, self.gear, self.brake, self.steer_past
