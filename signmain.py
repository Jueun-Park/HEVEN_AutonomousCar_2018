import signcros
import signuturn
from multiprocessing import Process
from Multiprocessing import go
import sign_cam2


def main() :

    signmain=Process(target=sign_cam2.main())
    U_Process = Process(target=signuturn.sign_camstart())

    #test = Process(target=go.GO())
    #Cros_Process=Process(target=signcros.sign_camstart())



if __name__ == "__main__":
    main()



