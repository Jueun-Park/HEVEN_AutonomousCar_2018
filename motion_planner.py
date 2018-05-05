# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array

import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import cv2
from parabola import Parabola
from lidar import Lidar
from lanecam import LaneCam
import time
import videostream

class MotionPlanner:
    OBSTACLE_RADIUS = 500  # 원일 경우 반지름, 사각형일 경우 한 변
    PARKING_RADIUS = 500
    RANGE = 110

    def __init__(self): #, lidar_instance, lanecam_instance, signcam_instance):
        self.lidar = Lidar() #lidar_instance
        self.lanecam = LaneCam() #lanecam_instance
        self.signcam = None #signcam_instance

        self.motion = None

        self.motion_planner_frame = videostream.VideoStream()
        self.parking_lidar = videostream.VideoStream()


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

    def getFrame(self):
        return self.lanecam.getFrame() + (self.motion_planner_frame.read(), self.parking_lidar.read(), )

    def motion_plan(self, mission_num):
        if mission_num == 0: self.lane_handling()
        # 남은 것: 유턴, 동적, 정지선
        elif mission_num == 1: self.parkingline_handling()
        elif mission_num == 4: self.static_obs_handling()

    def lane_handling(self):
        self.lanecam.default_loop()

        if self.lanecam.left_coefficients is not None and self.lanecam.right_coefficients is not None:
            path_coefficients = (self.lanecam.left_coefficients + self.lanecam.right_coefficients) / 2
            path = Parabola(*path_coefficients)

            self.motion = (0, (path.get_value(-10), path.get_derivative(-10), path.get_curvature(-10)), None)

        else:
            self.motion = (0, None, None)

    def static_obs_handling(self, is_lane_required):
        RAD = np.int32(self.OBSTACLE_RADIUS)
        AUX_RANGE = np.int32((180 - self.RANGE) / 2)

        previous_data = None
        previous_target = None

        lidar_raw_data = self.lidar.data_list
        current_frame = np.zeros((RAD, RAD * 2), np.uint8)

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
            cv2.circle(current_frame, tuple(point), 65, 255, -1)  # 캔버스에 점 찍기

        if is_lane_required:
            self.lanecam.default_loop()
            if self.lanecam.left_current_points is not None and self.lanecam.right_current_points is not None:
                pass

        data = np.zeros((self.RANGE + 1, 2), np.int)

        if current_frame is not None:
            self.path(drv.InOut(data), drv.In(RAD), drv.In(AUX_RANGE), drv.In(current_frame), drv.In(np.int32(RAD * 2)), block=(self.RANGE + 1,1,1))
            data_transposed = np.transpose(data)

            for i in range(0, self.RANGE + 1):
                x = RAD + int(round(data_transposed[1][i] * np.cos(np.radians(i + AUX_RANGE)))) - 1
                y = RAD - int(round(data_transposed[1][i] * np.sin(np.radians(i + AUX_RANGE)))) - 1
                cv2.line(current_frame, (RAD, RAD), (x, y), 255)

            color = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

            count = np.sum(data_transposed[0])

            if count <= self.RANGE - 1:
                relative_position = np.argwhere(data_transposed[0] == 0) - 90 + AUX_RANGE
                minimum_distance = int(min(abs(relative_position)))

                for i in range(0, len(relative_position)):
                    if abs(relative_position[i]) == minimum_distance:
                        target = int(90 - AUX_RANGE + relative_position[i])

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
                if previous_data is not None and abs(
                        previous_data[previous_target - AUX_RANGE][1] - data[target - AUX_RANGE][1]) <= 10:
                    target = previous_target

                x_target = RAD + int(data_transposed[1][int(target) - AUX_RANGE] * np.cos(np.radians(int(target)))) - 1
                y_target = RAD - int(data_transposed[1][int(target) - AUX_RANGE] * np.sin(np.radians(int(target)))) - 1
                cv2.line(color, (RAD, RAD), (x_target, y_target), (0, 0, 255), 2)

                self.motion = (4, (data_transposed[1][target - AUX_RANGE], target), None)

                previous_data = data
                previous_target = target

            else:
                x_target = RAD + int(100 * np.cos(np.radians(int(-target)))) - 1
                y_target = RAD - int(100 * np.sin(np.radians(int(-target)))) - 1
                cv2.line(color, (RAD, RAD), (x_target, y_target), (0, 0, 255), 2)

                self.motion = (4, (10, target), None)

            self.motion_planner_frame.write(color)

    def stopline_handling(self):
        pass

    def parkingline_handling(self):
        RAD = self.PARKING_RADIUS
        self.lanecam.parkingline_loop()
        parking_line = self.lanecam.parkingline_info

        if parking_line is not None:
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

            r = 0
            obstacle_detected = False

            while not obstacle_detected and r <= 300:
                temp_x = RAD + parking_line[0] + int(r * np.cos(parking_line[2]))
                temp_y = int(RAD - (parking_line[1] + r * np.sin(parking_line[2])))

                try:
                    if current_frame[temp_y][temp_x] != 0:
                        obstacle_detected = True

                except: pass

                r += 1

            cv2.line(current_frame, (RAD + parking_line[0] + int(10 * np.cos(parking_line[2])),
                                     int(RAD - (parking_line[1] + 10 * np.sin(parking_line[2])))),
                     (RAD + parking_line[0] + int(r * np.cos(parking_line[2])),
                      int(RAD - (parking_line[1] + r * np.sin(parking_line[2])))), 100, 3)

            if not obstacle_detected:
                self.motion = (
                1, True, (parking_line[0], parking_line[1], np.rad2deg(parking_line[3]), np.rad2deg(parking_line[4])))

            else:
                self.motion = (
                1, False, (parking_line[0], parking_line[1], np.rad2deg(parking_line[3]), np.rad2deg(parking_line[4])))

            self.parking_lidar.write(current_frame)

        else: self.motion = (1, False, None)

    def Uturn_handling(self):
        pass

    def moving_obs_handling(self):
        pass

    def stop(self):
        self.stop_fg = True
        self.lidar.stop()
        self.lanecam.stop()

        # pycuda dealloc
        global context
        context.pop()
        context = None
        from pycuda.tools import clear_context_caches
        clear_context_caches()
        # pycuda dealloc end


if __name__ == "__main__" :
    from monitor import Monitor
    motion_plan = MotionPlanner()
    monitor = Monitor()

    while True:
        motion_plan.parkingline_handling()
        monitor.show('parking', *motion_plan.getFrame())
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    motion_plan.stop()