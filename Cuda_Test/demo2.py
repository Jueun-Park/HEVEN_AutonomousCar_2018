import pycuda.autoinit
import pycuda.driver as cuda
import numpy
import easycuda as ezcuda
import time

from pycuda.compiler import SourceModule

doublify = """
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
"""
a = numpy.random.randn(400).astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
fin_doublify = numpy.empty_like(a)
func1 = ezcuda.getFunction("doublify", doublify)
start = time.time()
func1(a_gpu, block=(40, 20, 1))
end = time.time()
cuda.memcpy_dtoh(fin_doublify, a_gpu)
print(fin_doublify)
print(end-start)

# 걸린 시간 기록
# 0.0010979175567626953