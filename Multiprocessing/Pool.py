from multiprocessing import Pool
import time
import os
import math

def f(x):
    print("값", x, "에 대한 작업 Pid = ",os.getpid())
    time.sleep(1)# 한 프로세스가 멈추어도 다른 프로세스가 돌아가는지 확인하기 위해 넣은 값
    return x*x


def function_pool():
    p = Pool(10)  # 프로세스 x개  준비-> 많이 늘린다고 좋은것은 아님 실행을 통해 적절한 값을 찾아야함


    print(p.map(f, range(0, 10)))

    # mapping



if __name__ == '__main__':
    startTime = float(time.time())
    function_pool()
    endTime = float(time.time())
    print("총 작업 시간", (endTime - startTime))



