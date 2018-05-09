# 차량 조향/속도 제어 프로그램
# 박준혁
# input: communication.py, motion_planner.py, lane_cam.py 등에서 주는 차가 앞으로 가는 데 필요한 모든 정보
# output: 통신에 보낼 조향/속도 정보


# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
#           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}
# self.change_mission = { 0 : (미션 변경 X), 1 : default, 2 : obs}
# car_front = 0.28

import time
import math


class Control:
    car_front = 0.28

    def __init__(self):
        self.gear = 0
        self.speed = 0
        self.steer = 0
        self.brake = 0

        self.velocity = 0
        self.steer_past = 0

        self.mission_num = 0  # (일반 주행 모드)

        self.default_mode = 0
        self.obs_mode = 1

        self.change_mission = 0

        #######################################
        self.speed_platform = 0
        self.ENC1 = 0
        self.cross_track_error = 0
        self.linear = 0
        self.cul = 0
        self.parking_time1 = 0
        self.parking_time2 = 0
        self.place = 0
        self.park_position = 0
        self.park_theta = 0
        self.obs_exist = 0
        self.count = 0
        self.stop_line = 0
        self.obs_r = 0
        self.obs_theta = 0
        self.turn_distance = 0

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

        self.u_sit = 0
        self.p_sit = 0

        self.default_y_dis = 0.1  # (임의의 값 / 1m)
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
            if first is None:
                return
            self.cross_track_error = first[0] / 100
            self.linear = first[1]
            self.cul = first[2] / 100

            if self.default_mode == 0:
                self.__default__()

            elif self.default_mode == 1:
                self.__default2__()

        elif self.mission_num == 1:
            self.place = first
            if second is not None:
                self.park_position = second[1]
                self.park_theta = (second[2] - 90)

            self.__parking__()

        elif self.mission_num == 3:
            self.obs_exist = first

            self.__moving__()

        elif self.mission_num == 6:
            self.turn_distance = first / 100

            self.__turn__()

        elif self.mission_num == 7:
            self.stop_line = first / 100

            self.__cross__()

        else:
            self.obs_r = first[0] / 100
            self.obs_theta = first[1]

            self.__obs__()

    def write(self):
        return self.gear, self.speed, self.steer, self.brake

    def ch_mission(self):
        # 일회용 미션 함수의 종료를 알리는 변수
        # default, obs는 0을 반환
        # parking, uturn, moving_obs, cross는 1을 반환
        return self.change_mission

    def __default__(self):
        self.steer = 0
        self.speed = 108
        self.gear = 0
        self.brake = 0

        self.change_mission = 0

        self.tan_value = self.linear * (-1)
        self.theta_1 = math.degrees(math.atan(self.tan_value))

        k = 1
        if abs(self.theta_1) < 15 and abs(self.cross_track_error) < 0.27:
            k = 0.5

        if self.speed_platform == 0:
            self.theta_2 = 0
        else:
            self.velocity = (self.speed_platform * 100) / 3600
            self.theta_2 = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))

        steer_now = (self.theta_1 + self.theta_2)

        self.adjust = 0.3
        if steer_now > 18:
            self.adjust = 0.4

        steer_final = ((self.adjust * self.steer_past) + ((1 - self.adjust) * steer_now))

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
        self.speed = 108
        self.gear = 0
        self.brake = 0

        self.change_mission = 0

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

        if self.speed_platform == 0:
            self.theta_error = 0
        else:
            self.velocity = (self.speed_platform * 100) / 3600
            self.theta_error = math.degrees(math.atan((k * self.cross_track_error) / self.velocity))

        self.adjust = 0.3
        self.correction_default = 1

        steer_now = (self.theta_line + self.theta_error)
        steer_final = (self.adjust * self.steer_past) + (1 - self.adjust) * steer_now * 1.387

        self.steer = steer_final * 71 * self.correction_default

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746

    def __obs__(self):
        self.steer = 0
        self.gear = 0
        self.brake = 0

        if self.mission_num == 2:
            self.speed = 36
            self.correction = 1.3
            self.adjust = 0.10

        elif self.mission_num == 4:  # 실험값 보정하기
            self.speed = 18
            self.correction = 1.3
            self.adjust = 0.10

        elif self.mission_num == 5:  # 실험값 보정하기
            self.speed = 54
            self.correction = 1.3
            self.adjust = 0.3

        self.change_mission = 0

        cal_theta = math.radians(abs(self.obs_theta - 90))
        self.cos_theta = math.cos(cal_theta)
        self.sin_theta = math.sin(cal_theta)

        if (self.obs_theta - 90) == 0:
            self.theta_obs = 0

        elif self.obs_theta == -35:
            self.theta_obs = 27

        elif self.obs_theta == -145:
            self.theta_obs = -27

        else:
            if self.obs_mode == 0:
                self.cul_obs = (self.obs_r + 2.08 * self.cos_theta) / (2 * self.sin_theta)

                # k = math.sqrt( x_position ^ 2 + 1.04 ^ 2)

                self.theta_obs = math.degrees(math.atan(1.04 / (self.cul_obs + 0.4925)))  # 장애물 회피각 산출 코드

            elif self.obs_mode == 1:
                self.cul_obs = (self.obs_r + (2.08 * self.cos_theta)) / (2 * self.sin_theta)
                self.theta_cal = math.atan((1.04 + (self.obs_r * self.cos_theta)) / self.cul_obs)

                self.son_obs = (self.cul_obs * math.sin(self.theta_cal)) - (self.obs_r * self.cos_theta)
                self.mother_obs = (self.cul_obs * math.cos(self.theta_cal)) + 0.4925

                self.theta_obs = math.degrees(math.atan(abs(self.son_obs / self.mother_obs)))

        if (self.obs_theta - 90) > 0:
            self.theta_obs = self.theta_obs * (-1)

        steer_final = (self.adjust * self.steer_past) + (1 - self.adjust) * self.theta_obs * 1.387 * self.correction

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

        self.change_mission = 0

        if self.obs_exist is True:
            self.speed = 0
            self.brake = 60
            if self.count == 0:
                self.count += 1
        else:
            self.speed = 36

            if self.count > 0:
                self.change_mission = 1

    def __cross__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        self.change_mission = 0

        if abs(self.stop_line) < 1:  # 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if self.t1 == 0:
                self.t1 = time.time()
            self.t2 = time.time()

            if (self.t2 - self.t1) < 3.0:
                self.speed = 0
                self.brake = 60
            else:
                self.speed = 54
                self.change_mission = 1

    def __parking__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        self.change_mission = 0

        if self.p_sit == 0:
            if self.place is False:
                self.steer = 0
                self.speed = 36
                self.gear = 0
                self.brake = 0

            elif self.place is True:
                self.speed = 0
                self.brake = 60
                if self.speed_platform == 0:
                    self.go = self.park_position / 1.7
                    self.park_theta_edit = self.park_theta
                    self.p_sit = 1

        elif self.p_sit == 1:
            self.speed = 54
            if self.pt1 == 0:
                self.pt1 = self.ENC1
            self.pt2 = self.ENC1

            #############################################
            self.edit_enc = self.park_theta_edit / 3.33

            if self.park_theta_edit > 0:
                self.edit_enc = self.edit_enc * (-1)
            #############################################

            if (self.pt2 - self.pt1) < self.go + 10:
                self.steer = 0

            elif self.go + 10 <= (self.pt2 - self.pt1) < self.go + 210 + self.edit_enc:  # 회전 엔코더 량 : 200
                self.steer = 1970

            if (self.pt2 - self.pt1) >= self.go + 210 + self.edit_enc:
                self.steer = 0
                self.speed = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 2

        elif self.p_sit == 2:
            if self.pt3 == 0:
                self.pt3 = self.ENC1
            self.pt4 = self.ENC1

            if (self.pt4 - self.pt3) < 50:
                self.speed = 54
                self.steer = 0
                self.brake = 0

            if (self.pt4 - self.pt3) >= 50:
                self.steer = 0
                self.brake = 60
                self.speed = 0

                if self.speed_platform == 0:
                    self.p_sit = 3

        elif self.p_sit == 3:
            self.gear = 2
            self.speed = 0
            self.steer = 0
            self.brake = 0

            if self.parking_time1 == 0:
                self.parking_time1 = time.time()

            self.parking_time2 = time.time()

            if (self.parking_time2 - self.parking_time1) > 10:
                self.p_sit = 4

        elif self.p_sit == 4:
            self.gear = 2
            self.speed = 54
            self.brake = 0

            if self.pt5 == 0:
                self.pt5 = self.ENC1
            self.pt6 = self.ENC1

            if abs(self.pt6 - self.pt5) < 50:
                self.speed = 54
                self.steer = 0
                self.brake = 0

            if abs(self.pt6 - self.pt5) >= 50:
                self.speed = 0
                self.steer = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 5

        elif self.p_sit == 5:
            self.gear = 2
            self.speed = 54
            self.brake = 0

            if self.pt7 == 0:
                self.pt7 = self.ENC1
            self.pt8 = self.ENC1

            if abs(self.pt8 - self.pt7) < 200:
                self.speed = 54
                self.steer = 1970
                self.brake = 0

            if abs(self.pt8 - self.pt7) >= 200:
                self.speed = 0
                self.steer = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 6

        elif self.p_sit == 6:
            self.gear = 0
            self.speed = 54
            self.steer = 0
            self.brake = 0
            self.change_mission = 1

    def __turn__(self):
        self.steer = 0
        self.speed = 36
        self.gear = 0
        self.brake = 0

        self.change_mission = 0

        if self.u_sit == 0:
            if self.turn_distance < 4.5:
                self.steer = 0
                self.speed = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.u_sit = 2

        elif self.u_sit == 1:
            self.speed = 36
            if self.ct1 == 0:
                self.ct1 = self.ENC1[0]
            self.ct2 = self.ENC1[0]

            if (self.ct2 - self.ct1) < 665:
                self.steer = -1970

            elif (self.ct2 - self.ct1) >= 665:
                self.steer = 0
                self.speed = 0
                self.brake = 60

                if self.speed_platform == 0:
                    self.u_sit = 2

        elif self.u_sit == 2:
            if self.ct3 == 0:
                self.ct3 = self.ENC1[0]
            self.ct4 = self.ENC1[0]

            if (self.ct4 - self.ct3) < 175:
                self.speed = 36
                self.steer = 1970
                self.brake = 0

            if (self.ct4 - self.ct3) >= 175:
                self.steer = 0
                self.brake = 60
                self.speed = 0

                if self.speed_platform == 0:
                    self.u_sit = 3

        elif self.u_sit == 3:
            self.steer = 0
            self.speed = 36
            self.brake = 0
            self.change_mission = 1


control = Control()
control.mission(0, (0, 0, 1000000000), None)
control.ch_mission()
print(control.steer)
print(control.change_mission)
