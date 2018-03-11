from signcampackage import signcros,signuturn
import threading
from multiprocessing import Process
from Multiprocessing import go

def main() :
    U_Process = Process(target=signuturn.sign_camstart())
    Cros_Process=Process(target=signcros.sign_camstart())
    test = Process(target=go.GO())

    U_Process.start()

    Cros_Process.start()
    test.start()


if __name__ == "__main__":
    main()



