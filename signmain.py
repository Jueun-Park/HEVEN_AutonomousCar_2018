from multiprocessing import Process
import cv2
import signuturn
import signcros

def main() :

    U_Process=Process(target=signuturn.sign_camstart())
    Cros_Process=Process(target=signcros.sign_camstart())
    U_Process.start()
    Cros_Process.start()




if __name__ == "__main__":
    main()
