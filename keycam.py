import threading
import keyboard

class KeyCam:
    def __init__(self):
        self.mission_num = 0
        keyboard.on_press(self.keylook)

    def get_mission(self):
        return self.mission_num

    def keylook(self, e):
        c = e.name
        if c == '0': self.mission_num = 0
        elif c == '1': self.mission_num = 1
        elif c == '2': self.mission_num = 2
        elif c == '3': self.mission_num = 365432676
        elif c == '4': self.mission_num = 4
        elif c == '5': self.mission_num = 5
        elif c == '6': self.mission_num = 6
        elif c == '7': self.mission_num = 7


if __name__ == "__main__":
    import time
    k = KeyCam()
    keyboard.hook(k.keylook)
    while True:
        print(k.get_mission())