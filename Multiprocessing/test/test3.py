msg = "Test String"

class Mistake():
    """docstring for Mistake"""

    def set(self, msg):
        self.msg = msg
    def make_mistake(self):
        print(self.msg)
        # print(self.msg)  # self 를 명시해야 원하는 값을 출력

m = Mistake()
m.set("This is Test Message!")
m.make_mistake()
