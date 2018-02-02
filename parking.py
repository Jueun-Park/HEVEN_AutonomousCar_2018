import time
# 엔코더는 앞바퀴 왼쪽에 부탁되어 있음


class Parking:

    car_front = 0.28

    def __init__(self, mission, corner1, corner2, corner3, obs):
        self.obs_y = obs[1]/100
        self.pt1 = 0
        self.pt2 = 0
        self.pt3 = 0
        self.pt4 = 0

        self.gear = 0
        self.steer = 0
        self.speed = 0
        self.sit = 0
        self.place = 0

        self.st1 = 0
        self.st2 = 0

        self.corner1 = corner1[1]
        self.corner2 = corner2[1]
        self.corner3 = corner3[1]

        self.mission = mission

    def judge(self):
        if self.corner1 < self.obs_y < self.corner2:
            self.place = 1
        elif self.corner2 < self.obs_y < self.corner3:
            self.place = 2
        else:
            self.place = 0

    def parking(self):
        Enc = PlatformSerial()
        Enc._read()
        ENC = Enc.ENC1

        self.speed = 36
        if self.sit == 0:
            if self.place == 2:
                if self.corner1 < 0.1:
                    self.speed = 0
                    self.sit = 1

            if self.place == 1:
                if self.corner2 < 0.1:
                    self.speed = 0
                    self.sit = 1

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
                self.steer = 0
                self.gear = 0






