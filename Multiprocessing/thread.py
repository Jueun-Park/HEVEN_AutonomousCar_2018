from threading import Thread

def work(start, end, result):
    total = 0
    for i in range(start, end):
        total += 1
    result.append(total)
    return


if __name__ == "__main__":
    START, END = 0, 100000000
    result = list()
    th1 = Thread(target=work, args=(START, END//4, result))
    th2 = Thread(target=work, args=(END//4, END//2, result))
    th3 = Thread(target=work, args=(END//2, 3*END//4, result))
    th4 = Thread(target=work, args=(3*END//4, END, result))
    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th1.join()
    th2.join()
    th3.join()
    th4.join()

print(f"Result: {sum(result)}")