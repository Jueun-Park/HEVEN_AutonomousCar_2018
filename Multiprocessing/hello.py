import multiprocessing
import time
import os


def test(i) :
    time.sleep(1)
    print("hello~~"+str(i))
    os.system("cmd.exe /c start echo hello")#cmd창 띄우기
    time.sleep(1)


def main() :

    procs=[]

    for i in range(4) :
        p1=multiprocessing.Process(target=test,args=(i,))
        p2 = multiprocessing.Process(target=test, args=(i,))
        procs.append(p1)
        p1.start()
        p2.start()

    for p1 in procs :
        p1.join()
        p2.join()
if __name__=="__main__" :
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    #순차적으로 처리되지 않는경우가 보임
