import math


class MSteering:

    front_dis = 0.5  ## 임의로 거리 지정 (실험값 필요)
    car_front = 0.28
    car_dis = 0.78  ## front_dis + car_front
    velocity = 36

    def __init__(self, obs):
        self.gear = 0
        self.steer = 0
        self.steer_past = 0
        self.speed = self.velocity
        self.dis = obs[0]
        self.y = obs[1]

    def steer_m(self):
        tan_value = (abs(self.dis)/self.y)
        theta = math.degrees(math.atan(tan_value))

        if self.dis < 0:
            theta = theta * (-1)
        else:
            theta = theta

        steer_now = theta
        adjust = 0.8
        steer_final = adjust*steer_now + (1-adjust)*self.steer_past

        self.steer = steer_final

        self.steer_past = steer_final

        if self.steer > 1970:
            self.steer = 1970
            self.steer_past = 27.746
        elif self.steer < -1970:
            self.steer = -1970
            self.steer_past = -27.746
