import multiprocessing
import time
import os


def test(i) :
    time.sleep(2)
    print("hello~~"+str(i))
    os.system("cmd.exe /c start echo hello")#cmd창 띄우기
    time.sleep(2)


def main() :

    procs=[]

    for i in range(4) :
        p=multiprocessing.Process(target=test,args=(i,))
        procs.append(p)
        p.start()

    for p in procs :
        p.join()
if __name__=="__main__" :
    main()