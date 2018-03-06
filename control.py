# 엔코더 앞바퀴 왼쪽에 부착
# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
#           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

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

        self.t1 = 0
        self.t2 = 0

        self.ct1 = 0
        self.ct2 = 0

        self.ct3 = 0
        self.ct4 = 0

        self.ct5 = 0
        self.ct6 = 0

        self.st1 = 0
        self.st2 = 0

        self.pt1 = 0
        self.pt2 = 0
        self.pt3 = 0
        self.pt4 = 0

        self.usit = 0
        self.psit = 0

        self.mission_num = mission_num

        if self.mission_num == 0:
            self.cross_track_error = first/100
            self.linear = second

            self.__default__()

        elif self.mission_num == 2:
            self.obs_pos = first

            self.__obs__()

        elif self.mission_num == 4:
            self.obs_pos = first

            self.__obs__()

        elif self.mission_num == 5:
            self.obs_pos = first

            self.__obs__()

        elif self.mission_num == 1:
            self.corner = first
            self.place = second

            self.__parking__()

        elif self.mission_num == 3:
            self.obs_exist = first

            self.__moving__()

        elif self.mission_num == 6:
            self.obs_uturn = first

            self.__uturn__()

        else:
            self.stop_line = first/100

            self.__cross__()

    def __default__(self):
        self.steer = 0
        self.speed = 54
        self.gear = 0
        self.brake = 0

        self.steer_past = 0

        self.tan_value = self.linear * (-1)
        self.theta_1 = math.degrees(math.atan(self.tan_value))

        k = 1
        if abs(self.theta_1) < 15 and abs(self.cross_track_error) < 0.27:
            k = 0.5

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

        return self.steer, self.speed, self.gear, self.steer_past

    def __obs__(self):  # brake 값 집어넣어서 수정바람
        self.steer = 0
        self.speed = 54
        self.gear = 0
        self.brake = 0

        self.steer_past = 0

        self.dis = self.obs_pos[0]
        self.y = self.obs_pos[1]

        self.tan_value = (abs(self.dis) / self.y)
        self.theta_3 = math.degrees(math.atan(self.tan_value))

        if self.dis < 0:
            self.theta_3 = self.theta_3 * (-1)
        else:
            self.theta_3 = self.theta_3

        self.adjust = 0.1

        steer_now = self.theta_3
        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * steer_now)

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

        return self.steer, self.speed, self.gear, self.steer_past

    def __moving__(self):  # brake 값 집어넣어서 수정바람
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        if self.obs_exist is True:
            self.speed = 0
        else:
            self.speed = 36

        return self.steer, self.speed, self.gear

    def __cross__(self):  # brake 값 집어넣어서 수정바람
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        if abs(self.stop_line) < 1:  # 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if self.t1 == 0:
                self.t1 = time.time()
            self.t2 = time.time()

            if (self.t2 - self.t1) < 3.0:  # 3초간 정지, 실험을 통해 보정 필요
                self.speed = 0
            else:
                self.speed = 54

        return self.steer, self.speed, self.gear

    def __parking__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        self.corner1 = self.corner[0]
        self.corner2 = self.corner[1]
        self.corner3 = self.corner[2]

        # self._read()
        # ENC = Enc.ENC1

        if self.psit == 0:
            if self.place == 1:
                if self.corner1 < 0.1:
                    self.speed = 0
                    self.psit = 1

            if self.place == 2:
                if self.corner2 < 0.1:
                    self.speed = 0
                    self.psit = 1

        if self.psit == 1:
            if self.pt1 == 0:
                self.pt1 = ENC[0]
            self.pt2 = ENC[0]

            if (self.pt2 - self.pt1) < 172:
                self.steer = 1127
                self.gear = 0
            elif 172 <= (self.pt2 - self.pt1) < 372:
                self.steer = 0
                self.gear = 0

        if 360 < (self.pt2 - self.pt1) < 380:
            self.speed = 0
            self.psit = 2

            if self.st1 == 0:
                self.st1 = time.time()
            self.st2 = time.time()

            if (self.st2 - self.st1) < 10:
                self.speed = 0
                self.steer = 0
                self.gear = 0
            else:
                self.psit = 3

        if self.psit == 3:
            if self.pt3 == 0:
                self.pt3 = ENC[0]
            self.pt4 = ENC[0]

            if 0 <= (self.pt4 - self.pt3) < -200:
                self.steer = 0
                self.speed = 36
                self.gear = 1
            elif -200 <= (self.pt4 - self.pt3) < -372:
                self.steer = 1127
                self.speed = 36
                self.gear = 1
            elif (self.pt4 - self.pt3) >= -372:
                self.steer = 0
                self.speed = 36
                self.gear = 0

        return self.steer, self.speed, self.gear

    # 후에 대화를 통해서 보강

    def __uturn__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        self.obs_y = self.obs_uturn[1] / 100

        # self._read()
        # ENC = Enc.ENC1

        if abs(self.obs_y) < self.car_front:
            self.speed = 0

            if self.usit == 0:
                self.usit = 1

        if self.usit == 1:
            self.speed = 36
            if self.ct1 == 0:
                self.ct1 = ENC[0]
            self.ct2 = ENC[0]

            if (self.ct2 - self.ct1) < 100:
                self.steer = 0

            elif 100 <= (self.ct2 - self.ct1) < 574:
                self.steer = -1970

            if (self.ct2 - self.ct1) >= 574:
                self.steer = 0
                self.speed = 0
                self.brake = 60
                if self.speed_platform == 0:
                    self.usit = 2

        elif self.usit == 2:
            if self.ct3 == 0:
                self.ct3 = ENC[0]
            self.ct4 = ENC[0]

            if (self.ct4 - self.ct3) < 118:
                self.speed = 36
                self.steer = 1970
                self.brake = 0

            if (self.ct4 - self.ct3) >= 118:
                self.steer = 0
                self.brake = 60
                self.speed = 0
                if self.speed_platform == 0:
                    self.usit = 3

        elif self.usit == 3:
            self.steer = 0
            self.speed = 36
            self.brake = 0

        return self.steer, self.speed, self.gear
