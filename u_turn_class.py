class UTURN:

    car_front = 0.28

    def __init__(self, end_line):
        self.end_line = end_line/100
        self.ct1 = 0
        self.ct2 = 0
        self.gear = 0
        self.steer = 0
        self.speed = 0
        self.escape = 0
        self.sit = 0

    def findline(self):
        if abs(self.end_line) < self.car_front:
            self.speed = 0
            self.sit = 1

    def turnning(self):
        Enc = PlatformSerial()
        Enc._read()
        ENC = Enc.ENC1
        if self.sit == 1:
            if self.ct1 == 0:
                self.ct1 = ENC[0]
            self.ct2 = ENC[0]

            if (self.ct2 - self.ct1) < 1:
                self.steer = 0
            elif 1 <= (self.ct2 - self.ct1) < 5.573:
                self.steer = -1970
            elif 5.573 <= (self.ct2 - self.ct1) < 6.011:
                self.steer = 1970
            elif (self.ct2 - self.ct1) >= 6.011:
                self.steer = 0
                self.escape = 1
    ## 난 죄가 없다
