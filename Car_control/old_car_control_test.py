def __default2__(self, cross_track_error, linear, cul):
    gear = 0
    speed = 54
    brake = 0
    self.change_mission = 0

    tan_value_1 = abs(linear)
    theta_1 = math.atan(tan_value_1)

    default_y_dis = 0.1  # (임의의 값 / 1m)
    son = cul * math.sin(theta_1) - default_y_dis
    mother = cul * math.cos(theta_1) + cross_track_error + 0.4925

    tan_value_2 = abs(son / mother)
    theta_line = math.degrees(math.atan(tan_value_2))

    if linear > 0:
        theta_line = theta_line * (-1)

    k = 1
    if abs(theta_line) < 15 and abs(cross_track_error) < 0.27:
        k = 0.5

    if self.speed_platform == 0:
        theta_error = 0
    else:
        velocity = (self.speed_platform * 100) / 3600
        theta_error = math.degrees(math.atan((k * cross_track_error) / velocity))

    adjust = 0.4
    correction_default = 1

    steer_now = (theta_line + theta_error)

    if abs(steer_now) > 15:
        speed = 72

    steer_final = (adjust * self.steer_past) + (1 - adjust) * steer_now * 1.387

    self.steer_past = steer_final

    steer = steer_final * 71 * correction_default
    if steer > 1970:
        steer = 1970
        self.steer_past = 27.746
    elif steer < -1970:
        steer = -1970
        self.steer_past = -27.746

    self.gear = gear
    self.speed = speed
    self.steer = steer
    self.brake = brake

#################################################################################################

if obs_mode == 0:
    if obs_theta == -35:
        theta_obs = 27
    elif obs_theta == -145:
        theta_obs = -27
    else:
        car_circle = 1.387
        cul_obs = (obs_r + 2.08 * cos_theta) / (2 * sin_theta)
        # k = math.sqrt( x_position ^ 2 + 1.04 ^ 2)

        theta_obs = math.degrees(math.atan(1.04 / (cul_obs + 0.4925)))  # 장애물 회피각 산출 코드