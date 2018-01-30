global gear, b, sit, ENC1
b = [0, 0]
sit = 0
gear = 0
steer = 0
main_speed = 54
mission_speed = 36
car_front = 0.28
##  small_angle = 19.189
##  small_radius = 2.2346
##  one_turn = 1.70

################Function######################


def u_turn(end_line):  ## 후에 회의를 통해서 조정
    global steer, gear, b, sit, speed_Mission
    escape = 0
    gear = 0
    steer = 0
    speed_Mission = mission_speed
    if abs(end_line) < car_front:
        speed_Mission = 0
        sit = 1
    ###########################################
    if sit == 1:
        speed_Mission = 36
        if b[0] == 0:
            b[0][0] = ENC1[0]
            b[0][1] = ENC1[1]
        b[1][0] = ENC1[0]
        b[1][1] = ENC1[1]

        if (b[1][0] - b[0][0]) < 1:
            steer = 0
        elif 1 <= (b[1][0] - b[0][0]) < 5.573:
            steer = -1970
        elif 5.573 <= (b[1][0] - b[0][0]) < 6.011:
            steer = 1970
        elif (b[1][0] - b[0][0]) >= 6.011:
            steer = 0
    return steer, speed_Mission, gear
