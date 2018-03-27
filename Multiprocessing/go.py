from multiprocessing import Process
import os
def GO():
    print(os.getpid())
    #for i in range(0,1000) :
        #print('Hi')


    
def GO2():
    print(os.getpid())
    #for i in range(0,100) :
        #print('Hi2')
if __name__=='__main__':
    pr1 = Process(target=GO())
    pr2 = Process(target=GO2())

    #pr1.start()
    #pr2.start()


    #pr1.join()
    #pr2.join()






