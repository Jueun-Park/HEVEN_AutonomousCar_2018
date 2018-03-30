
from multiprocessing import Process
import os
def cam1() :

    #cam = cv2.VideoCapture('C:/Users/Administrator/Desktop/0507_one_lap_normal.mp4')
    #cam.set(3, 480)
    #cam.set(4, 270)
    print(os.getpid())
    """
    if (not cam.isOpened()):
        print("cam open failed")

    while True :
        s, img=cam.read()
        print(os.getpid())
        cv2.imshow('cam',img)
        if cv2.waitKey(30) & 0xff == 27:
            break
    #cam.release()
    #cv2.destroyAllWindows()
    cv2.waitKey(0)
    """

def cam2() :

    #cam2 = cv2.VideoCapture('C:/Users/Administrator/Desktop/0507_one_lap_normal.mp4')
    #cam2.set(3, 480)
    #cam2.set(4, 270)
    print(os.getpid())
    """
    if (not cam2.isOpened()):
        print("cam open failed")

    while True :
        s,img=cam2.read()
        print(os.getpid())
        cv2.imshow('cam2', img)

        if cv2.waitKey(30) & 0xff == 27:
            break
    #cam2.release()
    #cv2.destroyAllWindows()
    cv2.waitKey(0)
"""

if __name__ == "__main__" :




    cam1p=Process(target=cam1)
    #procs.append(Process(target=cam1()))
    cam2p=Process(target=cam2)
    #procs.append(Process(target=cam2()))




