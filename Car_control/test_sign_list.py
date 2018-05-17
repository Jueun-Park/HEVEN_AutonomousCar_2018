#sign = [1]

#if len(sign) == 0:
#    print("default")
#else:
#    print("mission_change")

class Test:

    def __init__(self):
        self.test1 = 0
        self.test2 = 0

    def get_test1(self):
        self.test1 = 1
        return self.test1

    def compile(self):
        self.test2 = (0, self.get_test1())


test = Test()
test.compile()
print(test.test2)
