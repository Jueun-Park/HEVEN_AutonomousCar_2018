# 경로 설정 프로그램
# 김홍빈
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array (for default driving)
#         그 외 미션 주행에 필요한 각 정보들


# modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
#           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import cv2
from parabola import Parabola
from lidar import Lidar
from lanecam import LaneCam
import time
import video_stream
from sign_cam_test import SignCam

class MotionPlanner:
    def __init__(self):
        self.lidar = Lidar()  # lidar_instance

        self.lanecam = LaneCam()  # lane_cam_instance
        self.signcam = SignCam()  # sign_cam_instance

        self.mission_num = 0

        self.lap_during_collision = 0
        self.lap_during_clear = 0
        self.mission_start_lap = 0

        self.previous_target = None
        self.previous_data = None

        self.motion_parameter = None

        self.motion_planner_frame = video_stream.VideoStream()
        self.parking_lidar = video_stream.VideoStream()
        self.moving_obs_frame = video_stream.VideoStream()
        self.uturn_frame = video_stream.VideoStream()
        self.windows_is = []

        # pycuda alloc
        drv.init()
        global context
        from pycuda.tools import make_default_context
        context = make_default_context()

        mod = SourceModule(r"""
                #include <stdio.h>
                #include <math.h>

                #define PI 3.14159265
                __global__ void detect(int data[][2], int* rad, int* range, unsigned char *frame, int *pcol) {
                        for(int r = 0; r < rad[0]; r++) {
                            const int thetaIdx = threadIdx.x;
                            const int theta = thetaIdx + range[0];
                            int x = rad[0] + int(r * cos(theta * PI/180)) - 1;
                            int y = rad[0] - int(r * sin(theta * PI/180)) - 1;

                            if (data[thetaIdx][0] == 0) data[thetaIdx][1] = r;
                            if (*(frame + y * *pcol + x) != 0) data[thetaIdx][0] = 1;
                        }
                }
                """)

        self.path = mod.get_function("detect")
        # pycuda alloc end

        time.sleep(2)

    def get_frame(self):
        lanecam_getFrame = self.lanecam.getFrame()
        # monitor 에서 사용 여부
        # window_is:
        # 차선인식 raw 화면, 차선인식 결과 화면, 주차공간 인식 화면, 정지선 인식 화면,
        # 장애물 미션 화면, 주차 공간 라이다 화면, 동적장애물 화면, 유턴 화면
        self.windows_is =                               [True, True, False, False, False, False, False, False]
        if self.mission_num == 0: self.windows_is =     [True, True, False, False, False, False, False, False]
        elif self.mission_num == 1: self.windows_is =   [False, False, True, False, False, False, False, False]
        elif self.mission_num == 2: self.windows_is =   [False, False, False, False, True, False, False, False]
        elif self.mission_num == 3: self.windows_is =   [False, False, False, False, False, False, True, False]
        elif self.mission_num == 4: self.windows_is =   [False, False, False, False, True, False, False, False]
        elif self.mission_num == 5: self.windows_is =   [False, False, False, False, True, False, False, False]
        elif self.mission_num == 6: self.windows_is =   [False, False, False, False, False, False, False, True]
        elif self.mission_num == 7: self.windows_is =   [False, False, False, True, False, False, False, False]

        return lanecam_getFrame + (self.motion_planner_frame.read(),
                                          self.parking_lidar.read(), self.moving_obs_frame.read(), self.uturn_frame.read())

    def get_motion_parameter(self):
        return self.motion_parameter  # motion parameter

    def plan_motion(self, control_status):
        # ------------------------------------- 미션 번호 변경과 탈출 -------------------------------------
        # modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
        #           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

        if self.mission_num == 0:
            self.signcam.detect_one_frame()
            self.mission_num = self.signcam.get_mission()

            if self.mission_num != 0:
                self.mission_start_lap = time.time()

        if self.mission_num == 1:
            if control_status[1] == 6:
                self.mission_num = 0

        elif self.mission_num == 3:
            if control_status[2] == 2:
                self.mission_num = 0

        elif self.mission_num == 6:
            if control_status[0] == 3:
                self.mission_num = 0

        elif self.mission_num == 7:
            if control_status[2] == 2:
                self.mission_num = 0

        # --------------------------------------- 미션 수행 ----------------------------------------
        # modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
        #           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}
        if self.mission_num == 0:
            self.lane_handling()

        elif self.mission_num == 1:
            self.parkingline_handling()

        elif self.mission_num == 3:
            self.moving_obs_handling()

        elif self.mission_num == 2:
            # 부채살 반경, 부채살 사잇각, 장애물 offset 크기, 차선 offset 크기, timeout 시간(초)
            self.static_obs_handling(300, 110, 65, 100, 3)

        elif self.mission_num == 4:
            self.static_obs_handling(300, 110, 65, 60, 1.5)

        elif self.mission_num == 5:
            self.static_obs_handling(300, 110, 70, 60, 1.5)

        elif self.mission_num == 6:
            self.Uturn_handling()

        elif self.mission_num == 7:
            self.stopline_handling()

    def lane_handling(self):
        self.lanecam.default_loop(0)

        if self.lanecam.left_coefficients is not None and self.lanecam.right_coefficients is not None:
            path_coefficients = (self.lanecam.left_coefficients + self.lanecam.right_coefficients) / 2
            path = Parabola(path_coefficients[2], path_coefficients[1], path_coefficients[0])

            self.motion_parameter = (0, (path.get_value(-10), path.get_derivative(-10), path.get_curvature(-10)), None)

        else:
            self.motion_parameter = (0, None, None)

    def static_obs_handling(self, radius, angle, obs_size, lane_size, timeout):
        self.lanecam.default_loop(1)
        left_lane_points = self.lanecam.left_current_points
        right_lane_points = self.lanecam.right_current_points

        RAD = np.int32(radius)
        AUX_RANGE = np.int32((180 - angle) / 2)

        lidar_raw_data = self.lidar.data_list
        current_frame = np.zeros((RAD, RAD * 2), np.uint8)

        points = np.full((361, 2), -1000, np.int)  # 점 찍을 좌표들을 담을 어레이 (x, y), 멀리 -1000 으로 채워둠.

        for theta in range(0, 361):
            r = lidar_raw_data[theta] / 10  # 차에서 장애물까지의 거리, 단위는 cm

            if 2 <= r:  # 라이다 바로 앞 1cm 의 노이즈는 무시

                # r-theta 를 x-y 로 바꿔서 (실제에서의 위치, 단위는 cm)
                x = -r * np.cos(np.radians(0.5 * theta))
                y = r * np.sin(np.radians(0.5 * theta))

                # 좌표 변환, 화면에서 보이는 좌표(왼쪽 위가 (0, 0))에 맞춰서 집어넣는다
                points[theta][0] = round(x) + RAD
                points[theta][1] = RAD - round(y)

        for point in points:  # 장애물들에 대하여
            cv2.circle(current_frame, tuple(point), obs_size, 255, -1)  # 캔버스에 점 찍기

        data_ = np.zeros((angle + 1, 2), np.int)

        if current_frame is not None:
            self.path(drv.InOut(data_), drv.In(RAD), drv.In(AUX_RANGE), drv.In(current_frame), drv.In(np.int32(RAD * 2)),
                      block=(angle + 1, 1, 1))

        count_ = np.sum(np.transpose(data_)[0])

        if count_ == 0:
            self.lap_during_clear = time.time()

        else:
            self.lap_during_collision = time.time()

        print("Last obstacle before: ", self.lap_during_clear - self.lap_during_collision)

        # 다음 세 가지 조건을 모두 만족하면 탈출한다:
        # 전방이 깨끗한 시간이 timeout 이상일 때
        # 장애물을 한 번이라도 만난 뒤에
        # 미션을 시작한 지 3초 이상 지난 뒤에 (표지판을 인식하고 미션을 수행하기 전 탈출하는 것을 방지)
        if self.lap_during_clear - self.lap_during_collision >= timeout and self.lap_during_collision != 0 and \
                time.time() - self.mission_start_lap > 3:
            print("Escape!")
            self.lap_during_clear = 0
            self.lap_during_collision = 0
            self.mission_start_lap = 0
            self.mission_num = 0

        if left_lane_points is not None:
            for i in range(0, len(left_lane_points)):
                if left_lane_points[i] != -1:
                    if lane_size != 0:
                        cv2.circle(current_frame, (RAD - left_lane_points[i], RAD - 30 * i), lane_size, 100, -1)

        if right_lane_points is not None:
            for i in range(0, len(right_lane_points)):
               if right_lane_points[i] != -1:
                   if lane_size != 0:
                        cv2.circle(current_frame, (RAD + 299 -  right_lane_points[i], RAD - 30 * i), lane_size, 100, -1)

        data = np.zeros((angle + 1, 2), np.int)

        color = None
        target = None

        if current_frame is not None:
            self.path(drv.InOut(data), drv.In(RAD), drv.In(AUX_RANGE), drv.In(current_frame), drv.In(np.int32(RAD * 2)),
                      block=(angle + 1, 1, 1))

            data_transposed = np.transpose(data)


            # 장애물에 부딫힌 곳까지 하얀 선 그리기
            for i in range(0, angle + 1):
                x = RAD + int(data_transposed[1][i] * np.cos(np.radians(i + AUX_RANGE))) - 1
                y = RAD - int(data_transposed[1][i] * np.sin(np.radians(i + AUX_RANGE))) - 1
                cv2.line(current_frame, (RAD, RAD), (x, y), 255)

            # 진행할 방향을 빨간색으로 표시하기 위해 흑백에서 BGR 로 변환
            color = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

            # count 는 장애물이 부딪힌 방향의 갯수를 의미
            count = np.sum(data_transposed[0])

            if count <= angle - 1:
                relative_position = np.argwhere(data_transposed[0] == 0) - 90 + AUX_RANGE
                minimum_distance = int(min(abs(relative_position)))

                for i in range(0, len(relative_position)):
                    if abs(relative_position[i]) == minimum_distance:
                        target = int(90 + relative_position[i])

            else:
                target = int(np.argmax(data_transposed[1]) + AUX_RANGE)

            if np.sum(data_transposed[1]) == 0:
                r = 0
                found = False
                while not found:
                    for theta in (AUX_RANGE, 180 - AUX_RANGE):
                        x = RAD + int(r * np.cos(np.radians(theta))) - 1
                        y = RAD - int(r * np.sin(np.radians(theta))) - 1

                        if current_frame[y][x] == 0:
                            found = True
                            target = -theta
                            break
                    r += 1

            if target >= 0:
                if self.previous_data is not None and abs(
                      self.previous_data[self.previous_target - AUX_RANGE][1] - data[target - AUX_RANGE][1]) <= 1 and \
                        data[target - AUX_RANGE][1] != RAD - 1:
                    target = self.previous_target

                x_target = RAD + int(data_transposed[1][int(target) - AUX_RANGE] * np.cos(np.radians(int(target))))
                y_target = RAD - int(data_transposed[1][int(target) - AUX_RANGE] * np.sin(np.radians(int(target))))
                cv2.line(color, (RAD, RAD), (x_target, y_target), (0, 0, 255), 2)

                self.motion_parameter = (self.mission_num, (data_transposed[1][target - AUX_RANGE], target), None)

                self.previous_data = data
                self.previous_target = target

            else:
                x_target = RAD + int(100 * np.cos(np.radians(int(-target)))) - 1
                y_target = RAD - int(100 * np.sin(np.radians(int(-target)))) - 1
                cv2.line(color, (RAD, RAD), (x_target, y_target), (0, 0, 255), 2)

                self.motion_parameter = (self.mission_num, (10, target), None)

            if color is None: return

        self.motion_planner_frame.write(color)

    def stopline_handling(self):
        self.lanecam.stopline_loop()
        self.motion_parameter = (7, self.lanecam.stopline_info, None)

    def parkingline_handling(self):
        RAD = 300
        self.lanecam.parkingline_loop()
        parking_line = self.lanecam.parkingline_info

        lidar_raw_data = self.lidar.data_list
        current_frame = np.zeros((RAD, RAD * 2), np.uint8)

        points = np.full((361, 2), -1000, np.int)  # 점 찍을 좌표들을 담을 어레이 (x, y), 멀리 -1000 으로 채워둠.

        for angle in range(0, 361):
            r = lidar_raw_data[angle] / 10  # 차에서 장애물까지의 거리, 단위는 cm

            if 2 <= r <= RAD + 50:  # 라이다 바로 앞 1cm 의 노이즈는 무시

                # r-theta 를 x-y 로 바꿔서 (실제에서의 위치, 단위는 cm)
                x = -r * np.cos(np.radians(0.5 * angle))
                y = r * np.sin(np.radians(0.5 * angle))

                # 좌표 변환, 화면에서 보이는 좌표(왼쪽 위가 (0, 0))에 맞춰서 집어넣는다
                points[angle][0] = round(x) + RAD
                points[angle][1] = RAD - round(y)

        for point in points:  # 장애물들에 대하여
            cv2.circle(current_frame, tuple(point), 30, 255, -1)  # 캔버스에 점 찍기

        if parking_line is not None:
            r = 0
            obstacle_detected = False

            while not obstacle_detected and r <= 300:
                temp_x = RAD + parking_line[0] + int(r * np.cos(parking_line[2]))
                temp_y = int(RAD - (parking_line[1] + r * np.sin(parking_line[2])))

                try:
                    if current_frame[temp_y][temp_x] != 0:
                        obstacle_detected = True

                except:
                    pass

                r += 1

            cv2.line(current_frame, (RAD + parking_line[0] + int(10 * np.cos(parking_line[2])),
                                     int(RAD - (parking_line[1] + 10 * np.sin(parking_line[2])))),
                     (RAD + parking_line[0] + int(r * np.cos(parking_line[2])),
                      int(RAD - (parking_line[1] + r * np.sin(parking_line[2])))), 100, 3)

            if not obstacle_detected:
                self.motion_parameter = (1, True, (parking_line[0], parking_line[1], np.rad2deg(parking_line[3])))

            else:
                self.motion_parameter = (1, False, (parking_line[0], parking_line[1], np.rad2deg(parking_line[3])))

        else:
            self.motion_parameter = (1, False, None)

        self.parking_lidar.write(current_frame)

    def Uturn_handling(self):
        self.lanecam.default_loop(0)

        right_lane = None
        if self.lanecam.right_coefficients is not None:
            right_coefficients = self.lanecam.right_coefficients
            right_lane = Parabola(right_coefficients[2], right_coefficients[1], right_coefficients[0])

        UTURN_RANGE = 20
        RAD = np.int32(500)
        AUX_RANGE = np.int32((180 - np.int32(UTURN_RANGE)) / 2)

        lidar_raw_data = self.lidar.data_list
        uturn_frame = np.zeros((RAD, RAD * 2), np.uint8)

        points = np.full((361, 2), -1000, np.int)  # 점 찍을 좌표들을 담을 어레이 (x, y), 멀리 -1000 으로 채워둠.

        for angle in range(0, 361):
            r = lidar_raw_data[angle] / 10  # 차에서 장애물까지의 거리, 단위는 cm

            if 2 <= r:  # 라이다 바로 앞 1cm 의 노이즈는 무시

                # r-theta 를 x-y 로 바꿔서 (실제에서의 위치, 단위는 cm)
                x = -r * np.cos(np.radians(0.5 * angle))
                y = r * np.sin(np.radians(0.5 * angle))

                # 좌표 변환, 화면에서 보이는 좌표(왼쪽 위가 (0, 0))에 맞춰서 집어넣는다
                points[angle][0] = round(x) + RAD
                points[angle][1] = RAD - round(y)

        for point in points:  # 장애물들에 대하여
            cv2.circle(uturn_frame, tuple(point), 15, 255, -1)  # 캔버스에 점 찍기

        data = np.zeros((UTURN_RANGE + 1, 2), np.int)

        minimum_dist = None

        if uturn_frame is not None:
            self.path(drv.InOut(data), drv.In(RAD), drv.In(AUX_RANGE), drv.In(uturn_frame),
                      drv.In(np.int32(RAD * 2)),
                      block=(UTURN_RANGE + 1, 1, 1))

            for i in range(0, UTURN_RANGE + 1):
                x = RAD + int(round(data[i][1] * np.cos(np.radians(i + AUX_RANGE)))) - 1
                y = RAD - int(round(data[i][1] * np.sin(np.radians(i + AUX_RANGE)))) - 1
                cv2.line(uturn_frame, (RAD, RAD), (x, y), 255)

            data_transposed = data.transpose()

            minimum_dist = np.min(data_transposed[1])

        if right_lane is None: return
        self.motion_parameter = (6, minimum_dist, (right_lane.get_value(-10), right_lane.get_derivative(-10)))
        self.uturn_frame.write(uturn_frame)

    def moving_obs_handling(self):
        self.lanecam.default_loop(0)
        path = None

        # 차선 보고 가다가 앞에 막히면
        if self.lanecam.left_coefficients is not None and self.lanecam.right_coefficients is not None:
            path_coefficients = (self.lanecam.left_coefficients + self.lanecam.right_coefficients) / 2
            path = Parabola(path_coefficients[2], path_coefficients[1], path_coefficients[0])

        MOVING_OBS_RANGE = 60
        RAD = np.int32(300)
        AUX_RANGE = np.int32((180 - np.int32(MOVING_OBS_RANGE)) / 2)

        lidar_raw_data = self.lidar.data_list
        moving_obs_frame = np.zeros((RAD, RAD * 2), np.uint8)

        points = np.full((361, 2), -1000, np.int)  # 점 찍을 좌표들을 담을 어레이 (x, y), 멀리 -1000 으로 채워둠.

        for angle in range(0, 361):
            r = lidar_raw_data[angle] / 10  # 차에서 장애물까지의 거리, 단위는 cm

            if 2 <= r:  # 라이다 바로 앞 1cm 의 노이즈는 무시

                # r-theta 를 x-y 로 바꿔서 (실제에서의 위치, 단위는 cm)
                x = -r * np.cos(np.radians(0.5 * angle))
                y = r * np.sin(np.radians(0.5 * angle))

                # 좌표 변환, 화면에서 보이는 좌표(왼쪽 위가 (0, 0))에 맞춰서 집어넣는다
                points[angle][0] = round(x) + RAD
                points[angle][1] = RAD - round(y)

        for point in points:  # 장애물들에 대하여
            cv2.circle(moving_obs_frame, tuple(point), 25, 255, -1)  # 캔버스에 점 찍기

        data = np.zeros((MOVING_OBS_RANGE + 1, 2), np.int)

        if moving_obs_frame is not None:
            self.path(drv.InOut(data), drv.In(RAD), drv.In(AUX_RANGE), drv.In(moving_obs_frame), drv.In(np.int32(RAD * 2)),
                      block=(MOVING_OBS_RANGE + 1, 1, 1))

            for i in range(0, MOVING_OBS_RANGE + 1):
                x = RAD + int(round(data[i][1] * np.cos(np.radians(i + AUX_RANGE)))) - 1
                y = RAD - int(round(data[i][1] * np.sin(np.radians(i + AUX_RANGE)))) - 1
                cv2.line(moving_obs_frame, (RAD, RAD), (x, y), 255)

            data_transposed = data.transpose()
            collision_count = np.sum(data_transposed[0])  # 막힌 부채살 개수
            minimum_dist = np.min(data_transposed[1])  # 막힌 부채살 중 가장 짧은 길이

            if collision_count > 50 and minimum_dist < 200:
                # 미션 번호, (이차곡선의 함수값, 미분값, 곡률), 가도 되는지 안 되는지
                self.motion_parameter = (3, (path.get_value(-10), path.get_derivative(-10), path.get_curvature(-10)), False)

            else:
                self.motion_parameter = (3, (path.get_value(-10), path.get_derivative(-10), path.get_curvature(-10)), True)

        self.moving_obs_frame.write(moving_obs_frame)

    def stop(self):
        self.stop_fg = True
        self.lidar.stop()
        self.lanecam.stop()
        #self.signcam.stop()

        # pycuda dealloc
        global context
        context.pop()
        context = None
        from pycuda.tools import clear_context_caches
        clear_context_caches()
        # pycuda dealloc end


if __name__ == "__main__":
    from monitor import Monitor
    motion_plan = MotionPlanner()
    monitor = Monitor()

    while True:
        motion_plan.plan_motion()

        monitor.show('parking', *motion_plan.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    motion_plan.stop()
