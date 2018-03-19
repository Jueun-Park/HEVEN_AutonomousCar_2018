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
import hogbinvision


def main() :


    U_Process = Process(target=signuturn.sign_camstart)
    Cros_Process=Process(target=signcros.sign_camstart)
    Mov_Process=Process(target=signmov.sign_camstart)
    Park_Process=Process(target=signpark.sign_camstart)
    static_Process=Process(target=signstatic.sign_camstart)
    narrow_Process=Process(target=signnarrow.sign_camstart)
    lane_detect_process = Process(target=hogbinvision.vision)

    U_Process.start()
    Cros_Process.start()
    Mov_Process.start()
    Park_Process.start()
    static_Process.start()
    narrow_Process.start()
    lane_detect_process.start()




if __name__ == "__main__":
    main()



