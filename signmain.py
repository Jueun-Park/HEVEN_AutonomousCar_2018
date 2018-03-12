import signcros
import signuturn
import signmov
from multiprocessing import Process
from Multiprocessing import go
import sign_cam2
import threading


def main() :


    U_Process = Process(target=signuturn.sign_camstart)
    Cros_Process=Process(target=signcros.sign_camstart)
    Mov_Process=Process(target=signmov.sign_camstart)

    U_Process.start()
    Cros_Process.start()
    Mov_Process.start()



if __name__ == "__main__":
    main()



