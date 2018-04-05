import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

import threading

class LaneCam:
    def __init__(self):
        pass

    def findCenterOfMass(self, src):
        sum_of_y_mass_coordinates = np.array(0, np.int32)
        num_of_mass_points = np.array(0, np.int32)
        src_row = np.array(len(src))
        src_col = np.array(len(src[0]))
        self.path(drv.Out(sum_of_y_mass_coordinates),
                  drv.Out(num_of_mass_points),
                  drv.In(src),
                  drv.In(src_row),
                  block=(len(src), 1, 1))
        sum_of_y_mass_coordinates = int(sum_of_y_mass_coordinates)
        num_of_mass_points = int(num_of_mass_points)
        if num_of_mass_points == 0:
            center_of_mass_y = int(len(src) / 2)

        else:
            center_of_mass_y = int(round(sum_of_y_mass_coordinates / num_of_mass_points))

        return center_of_mass_y

    def show_loop(self):
#pycuda alloc
        drv.init()
        global context
        from pycuda.tools import make_default_context
        context = make_default_context()

        mod = SourceModule(r"""
            #include <stdio.h>
            
            __global__ void hi(int *psum, int *pnum, 
                                unsigned int *data, int *size)
            {
                int y = threadIdx.x;
                for (int x = 0; x < *size; x++)
                    if ((data + y * *size)[x] == 255) {
                        *psum += y;
                        *pnum += 1;
                    }
                
                printf("%d\n", threadIdx.x);
            }
            """)
        self.path = mod.get_function("hi")
#pycuda alloc end

        data = np.zeros((500, 1000), np.uint8)
        while True:
            self.findCenterOfMass(data)
            break

#pycuda dealloc
        context.pop()
        context = None
        from pycuda.tools import clear_context_caches
        clear_context_caches()
#pycuda dealloc end


if __name__ == "__main__":
    lane_cam = LaneCam()
    t1 = threading.Thread(target=lane_cam.show_loop)
    t1.start()