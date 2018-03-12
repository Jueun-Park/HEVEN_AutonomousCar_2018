import signcros
import signuturn
import signmov
import signpark
import signstatic
import signnarrow

from multiprocessing import Process
from Multiprocessing import go
import sign_cam2
import threading


def main() :


    #U_Process = threading.Thread(target=signuturn.sign_camstart)
    Cros_Process=threading.Thread(target=signcros.sign_camstart)
    Mov_Process=threading.Thread(target=signmov.sign_camstart)
    #Park_Process=Process(target=signpark.sign_camstart)
    #static_Process=Process(target=signstatic.sign_camstart)
    #narrow_Process=Process(target=signnarrow.sign_camstart)

    #U_Process.start()
    Cros_Process.start()
    Mov_Process.start()
    #Park_Process.start()
    #static_Process.start()
    #narrow_Process.start()




if __name__ == "__main__":
    main()



