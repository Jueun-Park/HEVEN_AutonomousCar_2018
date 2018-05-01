# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array

import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import cv2
import threading
from parabola import Parabola
from lidar import Lidar
import time


class MotionPlanner():
    OBSTACLE_RADIUS = 500  # 원일 경우 반지름, 사각형일 경우 한 변

    def __init__(self, lidar_instance): #, lidar_instance, lanecam_instance, signcam_instance):
        self.lidar = lidar_instance #lidar_instance
        self.lanecam = None #lanecam_instance
        self.signcam = None #signcam_instance

        self.target_angle = None
        self.distance = None

    def loop(self):
    # pycuda alloc
        drv.init()
        global context
        from pycuda.tools import make_default_context
        context = make_default_context()

        mod = SourceModule(r"""
            #include <stdio.h>
            #include <math.h>

            #define PI 3.14159265
            __global__ void detect(int data[][2], int *rad, unsigned char *frame, int *pcol) {
                    for(int r = 0; r < rad[0]; r++) {
                        const int thetaIdx = threadIdx.x;
                        const int theta = thetaIdx + 30;
                        int x = rad[0] + int(r * cos(theta * PI/180)) - 1;
                        int y = rad[0] - int(r * sin(theta * PI/180)) - 1;

                        if (data[thetaIdx + 30][0] == 0) data[thetaIdx + 30][1] = r;
                        if (*(frame + y * *pcol + x) != 0) data[thetaIdx + 30][0] = 1;
                    }
            }
            """)

        path = mod.get_function("detect")
        # pycuda alloc end

        RAD = np.int32(self.OBSTACLE_RADIUS)

        previous_data = None
        previous_target = None

        while True:
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
                cv2.circle(current_frame, tuple(point), 55, 255, -1)  # 캔버스에 점 찍기

            data = np.zeros((121, 2), np.int)

            if current_frame is not None:
                path(drv.InOut(data), drv.In(RAD), drv.In(current_frame), drv.In(np.int32(RAD * 2)), block=(121,1,1))
                data_transposed = np.transpose(data)

                for i in range(0, 121):
                    x = RAD + int(round(data_transposed[1][i] * np.cos(np.radians(i + 30)))) - 1
                    y = RAD - int(round(data_transposed[1][i] * np.sin(np.radians(i + 30)))) - 1
                    cv2.line(current_frame, (RAD, RAD), (x, y), 255)

                color = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

                count = np.sum(data_transposed[0])

                if previous_data is not None and abs(previous_data[previous_target][1] - data[previous_target][1]) <= 3:
                    target = previous_target

                if count <= 119:
                    relative_position = np.argwhere(data_transposed[0] == 0) - 60
                    minimum_distance = int(min(abs(relative_position)))

                    for i in range(0, len(relative_position)):
                        if abs(relative_position[i]) == minimum_distance:
                            target = int(60 + relative_position[i])

                else:
                    target = int(np.argmax(data_transposed[1]))

                if np.sum(data_transposed[1]) == 0:
                    target = 90

                self.target_angle = target
                self.distance = data_transposed[1][target]

                x_target = RAD + int(data_transposed[1][int(target)] * np.cos(np.radians(int(target)))) - 1
                y_target = RAD - int(data_transposed[1][int(target)] * np.sin(np.radians(int(target)))) - 1
                cv2.line(color, (RAD, RAD), (x_target, y_target), (0, 0, 255), 2)

                previous_data = data
                previous_target = target

                cv2.imshow('lidar', color)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # pycuda dealloc
        context.pop()
        context = None
        from pycuda.tools import clear_context_caches
        clear_context_caches()
        # pycuda dealloc end

    def initiate(self):
        thread = threading.Thread(target=self.loop)
        thread.start()


if __name__ == "__main__" :
    lidar = Lidar()
    lidar.initiate()
    time.sleep(2)

    motion_plan = MotionPlanner(lidar)
    motion_plan.initiate()
