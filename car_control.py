# 차량 조향/속도 제어 프로그램
# 박준혁
# input: communication.py, motion_planner.py, lane_cam.py 등에서 주는 차가 앞으로 가는 데 필요한 모든 정보
# output: 통신에 보낼 조향/속도 정보


# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
#           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}
# car_front = 0.28
# 임의로 수정하지 말 것

import time
import math


class Control:

    def __init__(self):
        self.speed_platform = 0
        self.enc = 0

        self.gear = 0
        self.speed = 0
        self.steer = 0
        self.brake = 0

        self.deceleration_speed = 0
        self.deceleration_brake = 0
        self.deceleration_trigger = 0
        #######################################
        self.mission_num = 0  # DEFAULT 모드

        self.default_mode = 0  # 0번 주행 모드
        self.obs_mode = 0

        self.change_mission = 0

        self.sign_list = [0,0,0,0,0,0,0,0]
        self.sign_t1 = 0
        self.sign_t2 = 0

        self.o_t01 = 0
        self.o_t02 = 0

        self.o_t11 = 0
        self.o_t12 = 0

        self.o_t21 = 0
        self.o_t22 = 0
        #######################################
        self.steer_past = 0

        self.parking_time1 = 0
        self.parking_time2 = 0
        self.count = 0

        self.t1 = 0
        self.t2 = 0

        self.dt1 = 0
        self.dt2 = 0

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

    def mission(self, mission_num, first, second, trigger):
        self.set_mission(mission_num)
        self.do_mission(first, second)
        self.deceleration(mission_num, trigger)
        print("deceleration trigger: ", self.deceleration_trigger)

    def set_mission(self, mission_num):
        self.mission_num = mission_num

    def do_mission(self, first, second):
        if self.mission_num == 0:
            if first is None:
                return
            self.__default__(first[0] / 100, first[1])

        elif self.mission_num == 1:
            self.__parking__(first, second[3], second[0] / 100, second[1])

        elif self.mission_num == 3:
            if first is None:
                self.__moving__(None, None, False)
            else:
                self.__moving__(first[0] / 100, first[1], second)

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
        return self.gear, self.deceleration_speed, self.steer, self.deceleration_brake

    def get_status(self):
        return self.u_sit, self.p_sit, self.change_mission

    def ch_mission(self):
        # 일회용 미션 함수의 종료를 알리는 변수
        # default는 0을 반환
        # obs는 1을 반환
        # parking, uturn, moving_obs, cross는 2을 반환
        return self.change_mission

    def deceleration(self, mission_num, trigger):
        self.deceleration_trigger = trigger

        if mission_num == 0:
            self.deceleration_trigger = trigger
        else:
            self.deceleration_trigger = 0

        if self.deceleration_trigger == 0:
            self.deceleration_speed = self.speed
            self.deceleration_brake = self.brake

        elif self.deceleration_trigger == 1:
            self.deceleration_speed = 24
            self.deceleration_brake = 0

    def __default__(self, cross_track_error, linear):
        gear = 0
        brake = 0

        if self.dt1 == 0:
            self.dt1 = time.time()
        self.dt2 = time.time()

        if abs(self.dt2 - self.dt1) < 5:
            speed = 36
        elif abs(self.dt2 - self.dt1) >= 5:
            speed = 60

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

        if abs(steer_now) > 15:
            speed = 54

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

    def __obs__(self, obs_r, obs_theta):
        gear = 0
        brake = 0

        obs_mode = 0
        car_circle = 1

        if self.mission_num == 2:
            speed = 42
            correction = 1.6
            adjust = 0.05
            obs_mode = 0

        elif self.mission_num == 4:  # 실험값 보정하기
            speed = 24
            correction = 1.3
            adjust = 0.10
            obs_mode = 1

        elif self.mission_num == 5:  # 실험값 보정하기
            speed = 48
            correction = 1.5
            adjust = 0.25
            obs_mode = 2

        else:
            print("MISSION NUMBER ERROR")
            speed = 0
            correction = 0.0
            adjust = 0.0

        cal_theta = math.radians(abs(obs_theta - 90))
        cos_theta = math.cos(cal_theta)
        sin_theta = math.sin(cal_theta)

        if (90 - obs_theta) == 0:
            theta_obs = 0

        else:
            if obs_mode == 0:
                if self.o_t01 == 0:
                    self.o_t01 = time.time()
                self.o_t02 = time.time()

                if abs(self.o_t02 - self.o_t01) < 27:
                    self.change_mission = 1
                elif abs(self.o_t02 - self.o_t01) >= 27:
                    self.change_mission = 2

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

                    if abs(theta_obs) > 15:
                        speed = 12

            elif obs_mode == 1:
                if self.o_t11 == 0:
                    self.o_t11 = time.time()
                self.o_t12 = time.time()

                if abs(self.o_t12 - self.o_t11) < 36:
                    self.change_mission = 1
                elif abs(self.o_t12 - self.o_t11) >= 36:
                    self.change_mission = 2

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

                    if abs(theta_obs) > 15:
                        speed = 12

                #if obs_theta == -35:
                #    theta_obs = 27
                #elif obs_theta == -145:
                #    theta_obs = -27
                #else:
                #    car_circle = 1.387
                #    cul_obs = (obs_r + (2.08 * cos_theta)) / (2 * sin_theta)
                #    theta_cal = math.atan((1.04 + (obs_r * cos_theta)) / cul_obs) / 2

                #    son_obs = (cul_obs * math.sin(theta_cal)) - (obs_r * cos_theta)
                #    mother_obs = (cul_obs * math.cos(theta_cal)) + 0.4925

                #    theta_obs = math.degrees(math.atan(abs(son_obs / mother_obs)))

                #    if abs(theta_obs) > 15:
                #        speed = 9

            elif obs_mode == 2:
                if self.o_t21 == 0:
                    self.o_t21 = time.time()
                self.o_t22 = time.time()

                if abs(self.o_t22 - self.o_t21) < 27:
                    self.change_mission = 1
                elif abs(self.o_t22 - self.o_t21) >= 27:
                    self.change_mission = 2

                if obs_theta == -35:
                    theta_obs = 10
                    speed = 9
                elif obs_theta == -145:
                    theta_obs = -10
                    speed = 9
                else:
                    car_circle = 1.387
                    cul_obs = (obs_r + (2.08 * cos_theta)) / (2 * sin_theta)
                    theta_cal = math.atan((1.04 + (obs_r * cos_theta)) / cul_obs) / 2

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

    def __moving__(self, moving_error, moving_linear, obs_exist):
        gear = 0

        if moving_error is None:
            steer = 0
        else:
            ############################################################################################################
            tan_value = moving_linear * (-1)
            theta_1 = math.degrees(math.atan(tan_value))

            k = 0.5

            if self.speed_platform == 0:
                theta_2 = 0
            else:
                velocity = (self.speed_platform * 100) / 3600
                theta_2 = math.degrees(math.atan((k * moving_error) / velocity))

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
            ############################################################################################################

        self.change_mission = 0

        if obs_exist is False:
            speed = 0
            brake = 80
            if self.count == 0:
                self.count += 1
        else:
            if moving_error is None:
                speed = 12
            else:
                speed = 18
            brake = 0
            if self.count > 0:
                self.change_mission = 2

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

        if abs(stop_line) < 1.7:  # 기준선까지의 거리값, 경로생성 알고리즘에서 값 받아오기
            if self.t1 == 0:
                self.t1 = time.time()
            self.t2 = time.time()

            if (self.t2 - self.t1) < 4.0:
                speed = 0
                brake = 80
            else:
                speed = 54
                brake = 0
                self.change_mission = 2

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake

    def __parking__(self, place, park_position, park_error, park_linear):
        gear = 0
        speed = 36
        steer = 0
        brake = 0
        print(self.p_sit)

        self.change_mission = 0

        if self.p_sit == 0:
            if place is False:
                gear = 0
                speed = 36
                brake = 0

                tan_value = park_linear * (-1)
                theta_1 = math.degrees(math.atan(tan_value))

                k = 0.5

                if self.speed_platform == 0:
                    theta_2 = 0
                else:
                    velocity = (self.speed_platform * 100) / 3600
                    theta_2 = math.degrees(math.atan((k * park_error) / velocity))

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


            elif place is True:
                speed = 0
                brake = 60
                if self.speed_platform == 0:
                    self.go = park_position / 1.7
                    self.park_theta_edit = park_linear
                    self.p_sit = 1

        elif self.p_sit == 1:
            speed = 54
            if self.pt1 == 0:
                self.pt1 = self.enc
            self.pt2 = self.enc

            #############################################
            self.edit_enc = math.degrees(math.atan(abs(self.park_theta_edit))) / 3.33

            if self.park_theta_edit > 0:
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

            if (self.parking_time2 - self.parking_time1) > 11:
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
            self.change_mission = 2

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
            if turn_distance < 3.55:
                steer = 0
                speed = 0
                brake = 65

                if self.speed_platform == 0:
                    self.u_sit = 1

            else:
                steer = 0.5
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
            if self.ct5 == 0:
                self.ct5 = self.enc
            self.ct6 = self.enc

            if (self.ct6 - self.ct5) < 50:
                speed = 36
                steer = 0
                brake = 0

            if (self.ct6 - self.ct5) >= 50:
                speed = 0
                steer = 0
                brake = 70

                if self.speed_platform == 0:
                    self.u_sit = 4

        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake


if __name__ == '__main__':
    control = Control()
    control.mission(0, (0, 0, 1000000000), None)
    control.ch_mission()
    print(control.steer)
    print(control.change_mission)
