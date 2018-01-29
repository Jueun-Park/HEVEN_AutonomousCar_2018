from multiprocessing import Pool
import time
import os
import math

def f(x):
    print("값", x, "에 대한 작업 Pid = ",os.getpid())
    time.sleep(1)# 한 프로세스가 멈추어도 다른 프로세스가 돌아가는지 확인하기 위해 넣은 값
    return x*x

if __name__ == '__main__':
    p = Pool(3) # 프로세스 3개 준비
    startTime = int(time.time())
    print(p.map(f, range(0,10))) #mapping
    endTime = int(time.time())
    print("총 작업 시간", (endTime - startTime))