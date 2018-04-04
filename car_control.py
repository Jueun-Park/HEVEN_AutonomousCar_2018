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

        self.t1 = 0
        self.t2 = 0

        self.ct1 = 0
        self.ct2 = 0

        self.ct3 = 0
        self.ct4 = 0

        self.st1 = 0
        self.st2 = 0

        self.pt1 = 0
        self.pt2 = 0
        self.pt3 = 0
        self.pt4 = 0
        self.pt5 = 0
        self.pt6 = 0
        self.pt7 = 0
        self.pt8 = 0

        self.usit = 0
        self.psit = 0

        #######################################
        # communication.py 에서 데이터 받아오기#
        self.speed_platform = platform.speed_platform
        self.ENC1 = platform.ENC_with_time
        #######################################

        self.mission_num = mission_num

        if self.mission_num == 0:
            self.cross_track_error = first/100
            self.linear = second

            self.__default__()

        elif self.mission_num == 2:
            self.obs_r = first[0]
            self.obs_theta = first[1]

            self.__obs__()

        elif self.mission_num == 4:
            self.obs_theta = first
            self.rad = (second/100)

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

        self.tan_value = self.linear * (-1)
        self.theta_1 = math.degrees(math.atan(self.tan_value))

        k = 1
        if abs(self.theta_1) < 15 and abs(self.cross_track_error) < 0.27:
            k = 0.5

        self.velocity = (self.speed_platform*100)/3600

        self.theta_2 = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))

        self.adjust = 0.1

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

    def __obs__(self):
        self.steer = 0
        self.speed = 54
        self.gear = 0
        self.brake = 0

        cal_theta = abs(self.obs_theta)
        x_position = (self.rad + 2.08 * math.cos(cal_theta)) / (2 * math.sin(cal_theta))

        # k = math.sqrt( x_position ^ 2 + 1.04 ^ 2)

        self.theta_obs = math.degrees(math.atan(1.04 / (x_position + 0.4925))) * 1.387

        self.adjust = 0.1

        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * self.theta_obs)

        if self.obs_theta < 0:
            steer_final = steer_final * (-1)

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

        return self.steer, self.speed, self.gear, self.brake, self.steer_past

    def __moving__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        if self.obs_exist is True:
            self.speed = 0
            self.brake = 60
        else:
            self.speed = 36

        return self.steer, self.speed, self.gear

    def __cross__(self):
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
                self.brake = 60
            else:
                self.speed = 54

        return self.steer, self.speed, self.gear, self.brake

    def __parking__(self):
        self.steer = 0
        self.speed = 0
        self.gear = 0
        self.brake = 0

        self.parking_time1 = 0
        self.parking_time2 = 0

        # self.corner1 = self.corner[0]
        # self.corner2 = self.corner[1]
        # self.corner3 = self.corner[2]

        # self._read()
        # ENC = Enc.ENC1

        # 주차 매크로를 시작할 일정 거리까지 이동하는 코드 짜놓기 / 비전이랑 라이다 회의 필요

        if self.psit == 1:
            self.speed = 36
            if self.pt1 == 0:
                self.pt1 = self.ENC1[0]
            self.pt2 = self.ENC1[0]

            if (self.pt2 - self.pt1) < 100:
                self.steer = 0

            elif 100 <= (self.pt2 - self.pt1) < 280:  # 변경할 때 걸리는 엔코더 초과값 계산 및 보정 필요(593)
                self.steer = 1970

            if (self.pt2 - self.pt1) >= 290:
                self.steer = 0
                self.speed = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.psit = 2

        elif self.psit == 2:
            if self.pt3 == 0:
                self.pt3 = self.ENC1[0]
            self.pt4 = self.ENC1[0]

            if (self.pt4 - self.pt3) < 100:
                self.speed = 36
                self.steer = 0
                self.brake = 0

            if (self.pt4 - self.pt3) >= 100:
                self.steer = 0
                self.brake = 60
                self.speed = 0

                if self.speed_platform == 0:
                    self.psit = 3

        elif self.psit == 3:
            self.speed_for_write = 0
            self.steer_for_write = 0

            if self.parking_time1 == 0:
                self.parking_time1 = time.time()

            self.parking_time2 = time.time()

            if (self.parking_time2 - self.parking_time1) > 10:
                self.psit = 4

        elif self.psit == 4:
            self.gear = 1
            self.speed = 36
            self.brake = 0

            if self.pt5 == 0:
                self.pt5 = self.ENC1[0]
            self.pt6 = self.ENC1[0]

            if abs(self.pt6 - self.pt5) < 100:
                self.speed = 36
                self.steer = 0
                self.brake = 0

            if abs(self.pt6 - self.pt5) >= 100:
                self.steer = 0
                self.brake = 60
                self.speed = 0

                if self.speed_platform == 0:
                    self.psit = 5

        elif self.psit == 5:
            self.gear = 1
            self.speed = 36
            self.brake = 0

            if self.pt7 == 0:
                self.pt7 = self.ENC1[0]
            self.pt8 = self.ENC1[0]

            if abs(self.pt8 - self.pt7) < 185:
                self.speed = 36
                self.steer = 1970
                self.brake = 0

            if abs(self.pt8 - self.pt7) >= 185:
                self.steer = 0
                self.brake = 60
                self.speed = 0

                if self.speed_platform == 0:
                    self.psit = 6

        elif self.psit == 6:
            self.gear = 0
            self.speed = 36
            self.steer = 0
            self.brake = 0

        return self.steer, self.speed, self.gear, self.brake

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
                self.ct1 = self.ENC1[0]
            self.ct2 = self.ENC1[0]

            if (self.ct2 - self.ct1) < 100:
                self.steer = 0

            elif 100 <= (self.ct2 - self.ct1) < 730:
                self.steer = -1970

            if (self.ct2 - self.ct1) >= 730:
                self.steer = 0
                self.speed = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.usit = 2

        elif self.usit == 2:
            if self.ct3 == 0:
                self.ct3 = self.ENC1[0]
            self.ct4 = self.ENC1[0]

            if (self.ct4 - self.ct3) < 134:
                self.speed = 36
                self.steer = 1970
                self.brake = 0

            if (self.ct4 - self.ct3) >= 134:
                self.steer = 0
                self.brake = 60
                self.speed = 0

                if self.speed_platform == 0:
                    self.usit = 3

        elif self.usit == 3:
            self.steer = 0
            self.speed = 36
            self.brake = 0

        return self.steer, self.speed, self.gear, self.brake
