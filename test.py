import time


class Macro:

    def __init__(self):
        self.ct1 = 0
        self.ct2 = 0
        self.gear = 0
        self.steer = 0
        self.speed = 0
        self.sit = 0

        self.pt1 = 0
        self.pt2 = 0
        self.pt3 = 0
        self.pt4 = 0

        self.st1 = 0
        self.st2 = 0

        self.__macro__()

    def __macro__(self):
        self.speed = 36
        self.steer = 0
        self.gear = 0

        Enc = PlatformSerial()
        Enc._read()
        ENC = Enc.ENC1
        if self.sit == 0:
            self.speed = 36
            if self.ct1 == 0:
                self.ct1 = ENC[0]
            self.ct2 = ENC[0]

            if (self.ct2 - self.ct1) < 1:
                self.steer = 0
            elif 1 <= (self.ct2 - self.ct1) < 3.254:
                self.steer = -1970
            elif (self.ct2 - self.ct1) >= 3.254:
                self.steer = 0
                self.sit = 1

        self.speed = 36
        if self.sit == 1:
            if self.pt1 == 0:
                self.pt1 = ENC[0]
            self.pt2 = ENC[0]

            if (self.pt2 - self.pt1) < 1.724:
                self.steer = 1127
                self.gear = 0
            elif 1.724 <= (self.pt2 - self.pt1) < 3.724:
                self.steer = 0
                self.gear = 0

        if 3.6 < (self.pt2 - self.pt1) < 3.8:
            self.speed = 0
            self.sit = 2
            if self.st1 == 0:
                self.st1 = time.time()
            self.st2 = time.time()

            if (self.st2 - self.st1) < 10:
                self.speed = 0
            else:
                self.sit = 3

        if self.sit == 3:
            if self.pt3 == 0:
                self.pt3 = ENC[0]
            self.pt4 = ENC[0]

            if 0 <= (self.pt4 - self.pt3) < -2:
                self.steer = 0
                self.speed = 36
                self.gear = 1
            elif -2 <= (self.pt4 - self.pt3) < -3.724:
                self.steer = 1127
                self.gear = 1
            elif (self.pt4 - self.pt3) >= -3.724:
                self.steer = 3
                self.gear = 0

        return self.steer, self.speed, self.gear


