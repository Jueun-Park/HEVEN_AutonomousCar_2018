import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

data = np.arange(2 * 37).reshape((2, 37))
Rad = np.arange(1)
current_frame = np.arange(500 * 1000).reshape((500, 1000))

mod = SourceModule(r"""
    #include <stdio.h>

    __global__ void say_hi(int data[][37], int *Rad, unsigned char frame[][1000])
    {
            const int i = threadIdx.x;
            printf("%d: %d %d [%d]\n", i, data[0][i], data[1][i], Rad[0]);
    }
""")

path = mod.get_function("say_hi")

path(drv.InOut(data), drv.In(Rad), drv.In(current_frame),
     block=(37, 1, 1))
