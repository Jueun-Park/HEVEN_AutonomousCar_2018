import lidar

modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
         'MOVING_OBS': 3, 'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

def lidarstart():
    current_lidar = lidar.Lidar()
    current_lidar.initiate()
    current_lidar.set_mode(modes['NARROW'])

    while True:
        #print(current_lidar.get_data())
        current_lidar.plot_data()