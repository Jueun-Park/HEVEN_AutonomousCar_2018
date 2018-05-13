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
        self.speed_platform = 0
        self.enc = 0

        self.gear = 0
        self.speed = 0
        self.steer = 0
        self.brake = 0
        #######################################
        self.mission_num = 0  # DEFAULT 모드

        self.default_mode = 0  # 0번 주행 모드
        self.obs_mode = 0

        self.change_mission = 0
        #######################################
        self.steer_past = 0

        self.parking_time1 = 0
        self.parking_time2 = 0
        self.count = 0

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

    def read(self, speed, enc):
        #######################################
        # communication.py 에서 데이터 받아오기#
        self.speed_platform = speed
        self.enc = enc
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

            if self.default_mode == 0:
                self.__default__(first[0] / 100, first[1])

            elif self.default_mode == 1:
                self.__default2__(first[0] / 100, first[1], first[2] / 100)

        elif self.mission_num == 1:
            if second is None:
                self.__parking__(first, None, None)
            else:
                self.__parking__(first, second[1], 90 - second[2])

        elif self.mission_num == 3:
            if first is None:
                self.__moving__(False)
            else:
                self.__moving__(first)

        elif self.mission_num == 6:
            self.__turn__(first / 100)

        elif self.mission_num == 7:
            if first is None:
                self.__cross__(100)
            else:
                self.__cross__(first / 100)

        else:
            self.__obs__(first[0] / 100, first[1])

    def write(self):
        return self.gear, self.speed, self.steer, self.brake

    def get_status(self):
        return self.u_sit, self.p_sit

    def ch_mission(self):
        # 일회용 미션 함수의 종료를 알리는 변수
        # default, obs는 0을 반환
        # parking, uturn, moving_obs, cross는 1을 반환
        return self.change_mission

    def __default__(self, cross_track_error, linear):
        gear = 0
        speed = 108
        brake = 0
        self.change_mission = 0

        tan_value = linear * (-1)
        theta_1 = math.degrees(math.atan(tan_value))

        k = 1
        if abs(theta_1) < 15 and abs(cross_track_error) < 0.27:
            k = 0.5

        if self.speed_platform == 0:
            theta_2 = 0
        else:
            velocity = (self.speed_platform * 100) / 3600
            theta_2 = math.degrees(math.atan((k * cross_track_error) / velocity))

        steer_now = (theta_1 + theta_2)

        adjust = 0.3

        steer_final = ((adjust * self.steer_past) + ((1 - adjust) * steer_now))
        self.steer_past = steer_final

        steer = steer_final * 71
        if steer > 1970:
            steer = 1970
            self.steer_past = 27.746
        elif steer < -1970:
            steer = -1970
            self.steer_past = -27.746

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __default2__(self, cross_track_error, linear, cul):
        gear = 0
        speed = 108
        brake = 0
        self.change_mission = 0

        tan_value_1 = abs(linear)
        theta_1 = math.atan(tan_value_1)

        default_y_dis = 0.1  # (임의의 값 / 1m)
        son = cul * math.sin(theta_1) - default_y_dis
        mother = cul * math.cos(theta_1) + cross_track_error + 0.4925

        tan_value_2 = abs(son / mother)
        theta_line = math.degrees(math.atan(tan_value_2))

        if linear > 0:
            theta_line = theta_line * (-1)

        k = 1
        if abs(theta_line) < 15 and abs(cross_track_error) < 0.27:
            k = 0.5

        if self.speed_platform == 0:
            theta_error = 0
        else:
            velocity = (self.speed_platform * 100) / 3600
            theta_error = math.degrees(math.atan((k * cross_track_error) / velocity))

        adjust = 0.3
        correction_default = 1

        steer_now = (theta_line + theta_error)
        steer_final = (adjust * self.steer_past) + (1 - adjust) * steer_now * 1.387

        self.steer_past = steer_final

        steer = steer_final * 71 * correction_default
        if steer > 1970:
            steer = 1970
            self.steer_past = 27.746
        elif steer < -1970:
            steer = -1970
            self.steer_past = -27.746

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __obs__(self, obs_r, obs_theta):
        gear = 0
        brake = 0

        obs_mode = 0
        car_circle = 1

        if self.mission_num == 2:
            speed = 54
            correction = 1.4
            adjust = 0.05
            obs_mode = 1

        elif self.mission_num == 4:  # 실험값 보정하기
            speed = 18
            correction = 1.3
            adjust = 0.10
            obs_mode = 1

        elif self.mission_num == 5:  # 실험값 보정하기
            speed = 42
            correction = 1.3
            adjust = 0.05
            obs_mode = 2

        else:
            print("MISSION NUMBER ERROR")
            speed = 0
            correction = 0.0
            adjust = 0.0

        self.change_mission = 0

        cal_theta = math.radians(abs(obs_theta - 90))
        cos_theta = math.cos(cal_theta)
        sin_theta = math.sin(cal_theta)

        if (90 - obs_theta) == 0:
            theta_obs = 0

        else:
            if obs_mode == 0:
                if obs_theta == -35:
                    theta_obs = 27
                elif obs_theta == -145:
                    theta_obs = -27
                else:
                    car_circle = 1.387
                    cul_obs = (obs_r + 2.08 * cos_theta) / (2 * sin_theta)
                    # k = math.sqrt( x_position ^ 2 + 1.04 ^ 2)

                    theta_obs = math.degrees(math.atan(1.04 / (cul_obs + 0.4925)))  # 장애물 회피각 산출 코드

            elif obs_mode == 1:
                if obs_theta == -35:
                    theta_obs = 27
                elif obs_theta == -145:
                    theta_obs = -27
                else:
                    car_circle = 1.387
                    cul_obs = (obs_r + (2.08 * cos_theta)) / (2 * sin_theta)
                    theta_cal = math.atan((1.04 + (obs_r * cos_theta)) / cul_obs) / 2

                    son_obs = (cul_obs * math.sin(theta_cal)) - (obs_r * cos_theta)
                    mother_obs = (cul_obs * math.cos(theta_cal)) + 0.4925

                    theta_obs = math.degrees(math.atan(abs(son_obs / mother_obs)))

            elif obs_mode == 2:
                if obs_theta == -35:
                    theta_obs = 20
                    speed = 18
                elif obs_theta == -145:
                    theta_obs = -20
                    speed = 18
                else:
                    car_circle = 1.387
                    cul_obs = (obs_r + (2.08 * cos_theta)) / (2 * sin_theta)
                    theta_cal = math.atan((1.04 + (obs_r * cos_theta)) / cul_obs) / 4

                    son_obs = (cul_obs * math.sin(theta_cal)) - (obs_r * cos_theta)
                    mother_obs = (cul_obs * math.cos(theta_cal)) + 0.4925

                    theta_obs = math.degrees(math.atan(abs(son_obs / mother_obs)))
            else:
                print("OBS MODE ERROR")
                theta_obs = 0

        if (90 - obs_theta) < 0:
            theta_obs = theta_obs * (-1)

        steer_final = (adjust * self.steer_past) + (1 - adjust) * theta_obs * correction * car_circle

        self.steer_past = steer_final

        steer = steer_final * 71
        if steer > 1970:
            steer = 1970
            self.steer_past = 27.746
        elif steer < -1970:
            steer = -1970
            self.steer_past = -27.746

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __moving__(self, obs_exist):
        steer = 0
        gear = 0

        self.change_mission = 0

        if obs_exist is True:
            speed = 0
            brake = 60
            if self.count == 0:
                self.count += 1
        else:
            speed = 36
            brake = 0
            if self.count > 0:
                self.change_mission = 1

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __cross__(self, stop_line):
        steer = 0
        speed = 36
        gear = 0
        brake = 0

        self.change_mission = 0

        if abs(stop_line) < 1:  # 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if self.t1 == 0:
                self.t1 = time.time()
            self.t2 = time.time()

            if (self.t2 - self.t1) < 3.0:
                speed = 0
                brake = 60
            else:
                speed = 54
                brake = 0
                self.change_mission = 1

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __parking__(self, place, park_position, park_theta):
        gear = 0
        speed = 36
        steer = 0
        brake = 0

        self.change_mission = 0

        if self.p_sit == 0:
            if place is False:
                gear = 0
                speed = 36
                steer = 0
                brake = 0

            elif place is True:
                speed = 0
                brake = 60
                if self.speed_platform == 0:
                    self.go = park_position / 1.7
                    self.park_theta_edit = park_theta
                    self.p_sit = 1

        elif self.p_sit == 1:
            speed = 54
            if self.pt1 == 0:
                self.pt1 = self.enc
            self.pt2 = self.enc

            #############################################
            self.edit_enc = abs(self.park_theta_edit) / 3.33

            if self.park_theta_edit < 0:
                self.edit_enc = self.edit_enc * (-1)
            #############################################

            if (self.pt2 - self.pt1) < self.go + 5:
                steer = 0

            elif self.go + 5 <= (self.pt2 - self.pt1) < self.go + 215 + self.edit_enc:  # 회전 엔코더 량 : 200
                steer = 1970

            if (self.pt2 - self.pt1) >= self.go + 215 + self.edit_enc:
                speed = 0
                steer = 0
                brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 2

        elif self.p_sit == 2:
            if self.pt3 == 0:
                self.pt3 = self.enc
            self.pt4 = self.enc

            if (self.pt4 - self.pt3) < 50:
                speed = 54
                steer = 0
                brake = 0

            if (self.pt4 - self.pt3) >= 50:
                speed = 0
                steer = 0
                brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 3

        elif self.p_sit == 3:
            gear = 2
            speed = 0
            steer = 0
            brake = 0

            if self.parking_time1 == 0:
                self.parking_time1 = time.time()

            self.parking_time2 = time.time()

            if (self.parking_time2 - self.parking_time1) > 10:
                self.p_sit = 4

        elif self.p_sit == 4:
            gear = 2
            speed = 54
            brake = 0

            if self.pt5 == 0:
                self.pt5 = self.enc
            self.pt6 = self.enc

            if abs(self.pt6 - self.pt5) < 50:
                speed = 54
                steer = 0
                brake = 0

            if abs(self.pt6 - self.pt5) >= 50:
                speed = 0
                steer = 0
                brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 5

        elif self.p_sit == 5:
            gear = 2
            speed = 54
            brake = 0

            if self.pt7 == 0:
                self.pt7 = self.enc
            self.pt8 = self.enc

            if abs(self.pt8 - self.pt7) < 205:
                speed = 54
                steer = 1970
                brake = 0

            if abs(self.pt8 - self.pt7) >= 205:
                speed = 0
                steer = 0
                brake = 60

                if self.speed_platform == 0:
                    self.p_sit = 6

        elif self.p_sit == 6:
            gear = 0
            speed = 54
            steer = 0
            brake = 0
            self.change_mission = 1

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __turn__(self, turn_distance):
        gear = 0
        speed = 36
        steer = 0
        brake = 0

        self.change_mission = 0

        if self.u_sit == 0:
            if turn_distance < 3.6:
                steer = 0
                speed = 0
                brake = 60

                if self.speed_platform == 0:
                    self.u_sit = 1

            else:
                steer = 0
                speed = 36
                brake = 0
                gear = 0

        elif self.u_sit == 1:
            speed = 36
            if self.ct1 == 0:
                self.ct1 = self.enc
            self.ct2 = self.enc

            if (self.ct2 - self.ct1) < 665:
                steer = -1970

            elif (self.ct2 - self.ct1) >= 665:
                speed = 0
                steer = 0
                brake = 60

                if self.speed_platform == 0:
                    self.u_sit = 2

        elif self.u_sit == 2:
            if self.ct3 == 0:
                self.ct3 = self.enc
            self.ct4 = self.enc

            if (self.ct4 - self.ct3) < 175:
                speed = 36
                steer = 1970
                brake = 0

            if (self.ct4 - self.ct3) >= 175:
                speed = 0
                steer = 0
                brake = 60

                if self.speed_platform == 0:
                    self.u_sit = 3

        elif self.u_sit == 3:
            speed = 36
            steer = 0
            brake = 0
            self.change_mission = 1

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake


control = Control()
control.mission(0, (0, 0, 1000000000), None)
control.ch_mission()
print(control.steer)
print(control.change_mission)
