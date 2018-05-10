import threading
import keyboard

class KeyCam:
    def __init__(self):
        self.mission_num = 0
        self.stop_fg = False
        self.threading = threading.Thread(target=self.keylook)
        self.threading.start()

    def get_mission(self):
        return self.mission_num

    def keylook(self):
        while True:
            for i in range(0, 9):
                if keyboard.is_pressed(str(i)): self.mission_num = i; break
            if self.stop_fg is True: break

    def stop(self):
        self.stop_fg = True

if __name__ == "__main__":
    k = KeyCam()
    while True:
        n = k.get_mission()
        print(n)
        if n == 8: break
    k.stop()