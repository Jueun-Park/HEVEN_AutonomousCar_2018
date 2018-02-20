
from multiprocessing import Pool
def plus(x,y) :
    return x+y

if __name__ == '__main__' :

    p=Pool(10)
    y=10
    print (p.map(plus, (range(0,10),y)))