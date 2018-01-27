from multiprocessing import Pool
import time

start_time = time.time()


def fibo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)


def print_fibo(n):  # 피보나치 결과를 확인합니다.
    print(fibo(n))

def main() :
    num_list = [31, 32, 33, 34]

    pool = Pool(processes=4)  # 4개의 프로세스를 사용합니다.
    pool.map(print_fibo, num_list)  # pool에 일을 던져줍니다.

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()



