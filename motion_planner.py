# 경로 설정
# input: 1. numpy array (from lidar)
#        2. numpy array (from lane_cam)
# output: 차선 center 위치, 기울기, 곡률이 담긴 numpy array

from parabola import Parabola
import threading

modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,
         'MOVING_OBS': 3, 'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

LEFT_BOUNDARY = -25
RIGHT_BOUNDARY = 25
TOP_BOUNDARY = 200

class MotionPlanner:

    def __init__(self, lidar_instance, lanecam_instance, signcam_instance):
        self.lidar = lidar_instance
        self.lanecam = lanecam_instance
        self.signcam = signcam_instance

        self.mode = modes['DEFAULT']

        # motion_plan: [mission_num, required information]
        self.motion_plan = []

    def get_path(self):
        self.lanecam.set_mode(0)
        lane_status = self.lanecam.get_data()

        path = Parabola((lane_status[0][0] + lane_status[1][0]) / 2,
                        (lane_status[0][1] + lane_status[1][1]) / 2, (lane_status[0][2] + lane_status[1][2]) / 2)

        return path

    def get_obs_status(self, total_point_list, top_boundary):

        is_empty = True

        for area in total_point_list:
            if area:
                is_empty = False

                left_dist = min(area)[0] - LEFT_BOUNDARY
                right_dist = RIGHT_BOUNDARY - max(area)[0]

                if left_dist > right_dist:
                    steer_direction = -1
                    depth = RIGHT_BOUNDARY - min(area)[0]
                    y = min(area)[1]

                elif left_dist == right_dist:
                    steer_direction = 0
                    depth = 0
                    y = min(area)[1]

                else:
                    steer_direction = 1
                    depth = max(area)[0] - LEFT_BOUNDARY
                    y = max(area)[1]

                return (steer_direction * depth, y)

        if is_empty: return (0, top_boundary)

    def loop(self):
        measure_point = -25

        while True:
            temp = self.signcam.read()

            if temp:
                self.mode = temp
                self.lanecam.set_mode(temp)
                self.lidar.set_mode(temp)

            if self.mode == modes['DEFAULT']:
                path = self.get_path()
                self.motion_plan = [modes['DEFAULT'], path.get_value(measure_point), path.get_derivative(measure_point)]

            elif self.mode == modes['PARKING']:
                # self.motion_plan = [1, undetermined]
                pass

            elif self.mode == modes['STATIC_OBS']:
                lane_points = self.lanecam.get_data()
                obs_points = self.lidar.get_data()

                total_points = [lane_points[0] + obs_points[0],
                                lane_points[1] + obs_points[1], lane_points[2] + obs_points[2]]

                self.motion_plan = [modes['STATIC_OBS'], self.get_obs_status(total_points, 200)]

            elif self.mode == modes['MOVING_OBS']:
                obs_points = self.lidar.get_data()

                if obs_points: obs_exist = True
                else: obs_exist = False

                self.motion_plan = [modes['MOVING_OBS'], obs_exist]

            elif self.mode == modes['S_CURVE']:
                obs_points = self.lidar.get_data()
                self.motion_plan = [modes['S_CURVE'], self.get_obs_status(obs_points, 30)]

            elif self.mode == modes['NARROW']:
                obs_points = self.lidar.get_data()
                self.motion_plan = [modes['NARROW'], self.get_obs_status(obs_points, 200)]

            elif self.mode == modes['UTURN']:
                # self.motion_plan = [6, value]
                pass

            elif self.mode == modes['CROSS_WALK']:
                # self.motion_plan = [7, value]
                pass

    def initiate(self):
        t = threading.Thread(target = self.loop)
        t.start()

    def get_motion(self):
        return self.motion_plan

left_coeffs = [60, -0.05, 0.0005]
right_coeffs = [-60, -0.025, 0.00075]
