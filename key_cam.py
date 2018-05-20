# sign_cam.py 가 개발이 완료되지 않았을 때, 나머지 프로그램 테스트를 위한 코드
# 김진웅
# input: 키보드 숫자 인풋
# output: 현재 수행중인 미션 번호

import keyboard


class KeyCam:
    def __init__(self):
        self.mission_num = 0
        keyboard.on_press(self.key_look)

    def get_mission(self):
        return self.mission_num

    def key_look(self, e):
        c = e.name
        if c == '0': self.mission_num = 0
        elif c == '1': self.mission_num = 1
        elif c == '2': self.mission_num = 2
        elif c == '3': self.mission_num = 3
        elif c == '4': self.mission_num = 4
        elif c == '5': self.mission_num = 5
        elif c == '6': self.mission_num = 6
        elif c == '7': self.mission_num = 7


if __name__ == "__main__":
    k = KeyCam()
    keyboard.hook(k.key_look)
    while True:
        print(k.get_mission())
