import time

start_time = time.time()

def fibo(n): # 회귀적 피보나치 함수
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n-1) + fibo(n-2)

num_list = [31, 32, 33, 34]
result_list = []
for num in num_list:
    result_list.append(fibo(num))

print(result_list)
print("--- %s seconds ---" % (time.time() - start_time))