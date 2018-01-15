# demo2.py와 비교하기 위한 소스코드
from multiprocessing import Pool
import numpy
import time

def doublify(a): 
    a *= 2
    return a

if __name__=='__main__':
    a = numpy.random.randn(400).astype(numpy.float32)
    start = time.time()
    with Pool(10) as p:
        print(p.map(doublify, a))
    end = time.time()
    print(end-start)

# 걸린 시간 기록
# 0.3605329990386963
# 0.32848310470581055