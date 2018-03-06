class Moving:

    velocity = 54

    def __init__(self, mission, obs):
        self.gear = 0
        self.steer = 0
        self.speed = self.velocity
        self.obs = obs

        self.mission = mission

    def moving_obs_m(self):
        if self.obs is not None:
            self.speed = 0
        else:
            self.speed = self.velocity
