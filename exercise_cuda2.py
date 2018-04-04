import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

data = np.arange(2 * 37).reshape((2, 37))
Rad = np.arange(1)

current_frame = np.arange(500 * 1000).reshape((500, 1000)).astype(np.uint8)

mod = SourceModule(r"""
    #include <stdio.h>
    #include <math.h>

    #define PI 3.14159265
    __global__ void say_hi(int data[][37], int *Rad, unsigned char frame[][1000])
    {       
            int x =0;
            int y=0;
            printf("%d\n",frame[y][x]);
            for(int r = 0; r < Rad[0]; r++) {
                const int thetaIdx = threadIdx.x;
                const int theta = thetaIdx * 5;
                printf("%d\n",frame[y][x]);
                int x = Rad[0] + int(r * cos(theta * PI/180)) - 1;
                int y = Rad[0] - int(r * sin(theta * PI/180)) - 1;
                
                if (data[0][thetaIdx] == 0) data[1][thetaIdx] = r;
                if (frame[y][x] != 0) data[0][thetaIdx] = 1;
                
                
            }
            printf("%d\n",frame[y][x]);
            printf("%d complete\n", threadIdx.x);
    } 
""")

path = mod.get_function("say_hi")

path(drv.InOut(data), drv.In(Rad), drv.In(current_frame),
     block=(1, 1, 1))
# block에 1024 까지 가능

