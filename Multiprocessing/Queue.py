from multiprocessing import Process, Queue

def f(k):
    k.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    p1 = Process(target=f, args=(q,))
    p2 = Process(target=f, args=(q,))
    p1.start()
    p2.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p1.join()
    p2.join()



