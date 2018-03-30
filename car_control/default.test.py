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

        self.mission_num = mission_num

        ######################################
        ## communication.py에서 데이터 받아오기
        self.speed_platform = 0
        ######################################

        self.cross_track_error = first/100
        self.linear = second

        self.__default__()

    def __default__(self):
        self.steer = 0
        self.speed = 54
        self.gear = 0
        self.brake = 0

        self.tan_value = self.linear * (-1)
        self.theta_1 = math.degrees(math.atan(self.tan_value))

        k = 1
        if abs(self.theta_1) < 15 and abs(self.cross_track_error) < 0.27:
            k = 0.5

        self.velocity = (self.speed_platform*100)/3600

        self.theta_2 = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))

        self.adjust = 0.3

        steer_now = (self.theta_1 + self.theta_2)
        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * steer_now)

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

        return self.steer, self.speed, self.gear, self.brake, self.steer_past
