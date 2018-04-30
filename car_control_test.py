# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
#           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

import time
import math


class Control:

    car_front = 0.28  # 수정 바람 - 차량 정지 시간

    def __init__(self):
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

        self.usit = 1
        self.psit = 1

        self.mission_num = 0  # (일반 주행 모드)

        self.mode = 0
        self.default_y_dis = 0.1  # (임의의 값 / 1m)

        #######################################
        self.speed_platform = 0
        self.ENC1 = 0
        self.cross_track_error = 0
        self.linear = 0
        self.cul = 0
        self.parking_time1 = 0
        self.parking_time2 = 0
        self.corner = 0
        self.place = 0
        self.obs_exist = 0
        self.obs_uturn = 0
        self.stop_line = 0
        self.obs_r = 0
        self.obs_theta = 0
        #######################################

    def read(self, speed, enc):
        #######################################
        # communication.py 에서 데이터 받아오기#
        self.speed_platform = speed
        self.ENC1 = enc
        #######################################

    def mission(self, mission_num, first, second):
        self.set_mission(mission_num)
        self.do_mission(first, second)

    def set_mission(self, mission_num):
        self.mission_num = mission_num

    def do_mission(self, first, second):

        if self.mission_num == 0:
            self.cross_track_error = first[0] / 100
            self.linear = first[1]
            self.cul = first[2] / 100

            if self.mode == 0:
                self.__default__()

            elif self.mode == 1:
                self.__default2__()

        elif self.mission_num == 1:
            # self.corner = first
            # self.place = second

            self.__parking__()

        elif self.mission_num == 3:
            self.obs_exist = first

            self.__moving__()

        elif self.mission_num == 6:
            self.obs_uturn = first

            self.__uturn__()

        elif self.mission_num == 7:
            self.stop_line = first/100

            self.__cross__()

        else:
            self.obs_r = first[0] / 100
            self.obs_theta = first[1]

            self.__obs__()

    def write(self):
        return self.gear, self.speed, self.steer, self.brake

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

    def __default2__(self):
        self.steer = 0
        self.speed = 54
        self.gear = 0
        self.brake = 0

        self.tan_value_1 = abs(self.linear)
        self.theta_1 = math.atan(self.tan_value_1)

        self.son = self.cul * math.sin(self.theta_1) - self.default_y_dis
        self.mother = self.cul * math.cos(self.theta_1) + self.cross_track_error + 0.4925

        self.tan_value_2 = abs(self.son / self.mother)
        self.theta_line = math.degrees(math.atan(self.tan_value_2))

        if self.linear > 0:
            self.theta_line = self.theta_line * (-1)

        k = 1
        if abs(self.theta_line) < 15 and abs(self.cross_track_error) < 0.27:
            k = 0.5

        self.velocity = (self.speed_platform * 100) / 3600

        self.theta_error = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))

        self.adjust = 0.1

        steer_now = (self.theta_line + self.theta_error)
        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * steer_now) * 1.387

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

    def __obs__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        cal_theta = math.radians(abs(self.obs_theta - 90))
        self.costheta = math.cos(cal_theta)
        self.sintheta = math.sin(cal_theta)

        if cal_theta == 0:
            self.theta_obs = 0

        else:
            self.cul_obs = (self.obs_r + 2.08 * self.costheta) / (2 * self.sintheta)

            # k = math.sqrt( x_position ^ 2 + 1.04 ^ 2)

            self.theta_obs = math.degrees(math.atan(1.04 / (self.cul_obs + 0.4925)))  # 장애물 회피각 산출 코드

            # self.theta_cal = math.asin((1.04 + self.obs_r * self.costheta) / self.cul_obs)

            # self.son_obs = self.cul_obs * math.sin(self.theta_cal) - self.obs_r * self.costheta
            # self.mother_obs = self.cul_obs * math.cos(self.theta_cal) + 0.4925

            # self.theta_obs = math.degrees(math.atan(abs(self.son_obs / self.mother_obs)))

        if (self.obs_theta - 90) > 0:
            self.theta_obs = self.theta_obs * (-1)

        self.adjust = 0.05

        steer_final = (self.adjust * self.steer_past) + ((1 - self.adjust) * self.theta_obs) * 1.387

        self.steer = steer_final * 71

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

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

    def __parking__(self):
        self.steer = 0
        self.speed = 0
        self.gear = 0
        self.brake = 0

        # self.corner1 = self.corner[0]
        # self.corner2 = self.corner[1]
        # self.corner3 = self.corner[2]

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

    def __uturn__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        self.obs_y = self.obs_uturn[1] / 100

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
