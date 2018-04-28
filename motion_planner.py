# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

import cv2
import threading
from lidar import Lidar
from lanecam import LaneCam

np.set_printoptions(linewidth=1000)

class MotionPlanner():
    def __init__(self):
        self.lidar = Lidar()
        self.lidar.initiate()


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
            __global__ void detect(int data[][2], int *rad, unsigned char frame[][1000])
            {
                    for(int r = 0; r < rad[0] - 2; r++) {
                        const int thetaIdx = threadIdx.x;
                        const int theta = thetaIdx;
                        int x = rad[0] + int(r * cos(theta * PI/180)) - 1;
                        int y = rad[0] - int(r * sin(theta * PI/180)) - 1;

                        if (data[thetaIdx][0] == 0) data[thetaIdx][1] = r;
                        if (frame[y][x] != 0) data[thetaIdx][0] = 1;
                    }
            }
            """)
        path = mod.get_function("detect")
        # pycuda alloc end

        Rad = np.int32(self.lidar.RADIUS)

        while True:
            data = np.zeros((181, 2), np.int)
            current_frame = self.lidar.frame

            if current_frame is not None:
                path(drv.InOut(data), drv.In(Rad), drv.In(current_frame), block=(181,1,1))
                data_transpose = np.transpose(data)

                for i in range(0, 181):
                    x = Rad + int(round(data_transpose[1][i] * np.cos(np.radians(i)))) - 1
                    y = Rad - int(round(data_transpose[1][i] * np.sin(np.radians(i)))) - 1
                    cv2.line(current_frame, (Rad, Rad), (x, y), 255)

                color = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

                count = np.sum(data_transpose[0])


                if count <= 179:
                    relative_position = np.argwhere(data_transpose[0] == 0) - 90
                    minimum_distance = int(min(abs(relative_position)))

                    for i in range(0, len(relative_position)):
                        if abs(relative_position[i]) == minimum_distance:
                            target = 90 + relative_position[i]

                else:
                    target = np.argmax(data_transpose[1])

                x_target = Rad + int(data_transpose[1][int(target)] * np.cos(np.radians(int(target)))) - 1
                y_target = Rad - int(data_transpose[1][int(target)] * np.sin(np.radians(int(target)))) - 1
                cv2.line(color, (Rad, Rad), (x_target, y_target), (0, 0, 255), 2)

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
    motion_plan = MotionPlanner()
    motion_plan.initiate()
