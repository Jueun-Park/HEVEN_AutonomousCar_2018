class GlobalU:
    ct1 = 0
    ct2 = 0
    gear = 0
    steer = 0
    speed = 36
    escape = 0
    sit = 0


class UTURN(GlobalU):

    car_front = 0.28

    def __init__(self, end_line):
        self.end_line = end_line/100

    def findline(self):
        if abs(self.end_line) < self.car_front:
            GlobalU.speed = 0
            GlobalU.sit = 1

    def turnning(self):
        Enc = PlatformSerial()
        Enc._read()
        ENC = Enc.ENC1
        if GlobalU.sit == 1:
            if GlobalU.ct1 == 0:
                GlobalU.ct1 = ENC[0]
            GlobalU.ct2 = ENC[0]

            if (GlobalU.ct2 - GlobalU.ct1) < 1:
                GlobalU.steer = 0
            elif 1 <= (GlobalU.ct2 - GlobalU.ct1) < 5.50:
                GlobalU.steer = -1970
            elif 5.50 <= (GlobalU.ct2 - GlobalU.ct1) < 6:
                GlobalU.steer = 1970
            elif (GlobalU.ct2 - GlobalU.ct1) >= 6:
                GlobalU.steer = 0
                GlobalU.escape = 1
