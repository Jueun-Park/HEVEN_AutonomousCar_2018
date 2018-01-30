class Moving:

    velocity = 36

    def __init__(self, mission,obs):
        self.gear = 0
        self.steer = 0
        self.speed = self.velocity
        self.dis = obs[0]
        self.y = obs[1]

        self.mission = mission

    def moving_obs_m(self):
        if abs(self.dis) > 0 and self.y < 1:
            self.speed = 0
        else:
            self.speed = self.velocity