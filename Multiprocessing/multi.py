from multiprocessing import Process, Queue

def work(start, end, result):
    total = 0
    for i in range(start, end):
        total += 1
    result.put(total)
    return

if __name__ == "__main__":
    START, END = 0, 100000000
    result = Queue()
    pr1 = Process(target=work, args=(START, END//2, result))
    pr2 = Process(target=work, args=(END//2, END, result))
    pr1.start()
    pr2.start()
    pr1.join()
    pr2.join()
    result.put('STOP')
    total = 0
    while True:
        tmp = result.get()
        if tmp == 'STOP':
            break
        else:
            total += tmp
    print(f"Result: {total}")