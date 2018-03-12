from multiprocessing import Process
import os
def GO():
    for i in range(0,55) :
        print('Hi\n')
    
def GO2():
    for i in range(0,200) :
        print('Hi2')
if __name__=='__main__':
    pr1 = Process(target=GO)
    pr2 = Process(target=GO2)

    pr1.start()
    pr2.start()


    pr1.join()
    pr2.join()