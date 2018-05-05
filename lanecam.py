import cv2
import numpy as np
import random
import time
import videostream


class LaneCam:
    # 웹캠 왜곡 보정에 필요한 값들
    camera_matrix_L = np.array([[474.383699, 0, 404.369647], [0, 478.128447, 212.932297], [0, 0, 1]])
    camera_matrix_R = np.array([[473.334870, 0, 386.312394], [0, 476.881433, 201.662339], [0, 0, 1]])
    distortion_coefficients_L = np.array([0.164159, -0.193892, -0.002730, -0.001859])
    distortion_coefficients_R = np.array([0.116554, -0.155379, -0.001045, -0.001512])

    # Bird eye view 에 필요한 값들
    pts1_L = np.float32([[0, 0], [0, 428], [780, 0], [780, 428]])
    pts2_L = np.float32([[0, 399], [440, 535], [527, 0], [560, 441]])
    pts1_R = np.float32([[0, 0], [0, 428], [780, 0], [780, 428]])
    pts2_R = np.float32([[5, 0], [-2, 384], [503, 324], [119, 471]])

    Bird_view_matrix_L = cv2.getPerspectiveTransform(pts1_L, pts2_L)
    Bird_view_matrix_R = cv2.getPerspectiveTransform(pts1_R, pts2_R)

    crop_L = [[210, 510], [262, 562]]
    crop_R = [[151, 451], [0, 300]]
    crop_R_parking = [[252, 452], [0, 503]]
    output_L = (563, 511)
    output_R = (503, 452)
    xreadParam_L = crop_L, camera_matrix_L, distortion_coefficients_L, Bird_view_matrix_L, output_L
    xreadParam_R = crop_R, camera_matrix_R, distortion_coefficients_R, Bird_view_matrix_R, output_R
    xreadparam_R_parking = crop_R_parking, camera_matrix_R, distortion_coefficients_R, Bird_view_matrix_R, output_R

    # HSV 을 이용한 차선 추출에 필요한 값들
    lower_yellow = np.array([0, 70, 120], dtype=np.uint8)
    upper_yellow = np.array([179, 255, 255], dtype=np.uint8)

    lower_white = np.array([187, 193, 187], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # 질량 중심 찾기 박스 너비
    BOX_WIDTH = 10

    def __init__(self):
        # 웹캠 2대 열기 # 양쪽 웹캠의 해상도를 800x448로 설정
        self.video_left = videostream.WebcamVideoStream(1, 800, 448)
        self.video_right = videostream.WebcamVideoStream(0, 800, 448)
        self.video_left.start()
        self.video_right.start()

        self.lane_cam_raw_frame = videostream.VideoStream()
        self.lane_cam_frame = videostream.VideoStream()
        self.parkingline_frame = videostream.VideoStream()


        # 현재 읽어온 프레임이 실시간으로 업데이트됌
        self.left_frame = None
        self.right_frame = None

        # 이전 프레임의 ROI 위치를 저장함
        self.left_previous_points = None
        self.right_previous_points = None

        # 현재 프레임의 ROI 위치를 저장함
        self.left_current_points = np.array([0] * 10)
        self.right_current_points = np.array([0] * 10)

        self.filtered_both = None

        # 차선과 닮은 이차함수의 계수 세 개를 담을 변수
        self.left_coefficients = None
        self.right_coefficients = None

        # 정지선 정보를 담을 변수
        self.stopline_info = None

        # 주차 공간 정보를 담을 변수
        self.parkingline_info = None

        time.sleep(1)

    def getFrame(self):
        return (self.lane_cam_raw_frame.read(), self.lane_cam_frame.read(), self.parkingline_frame.read())

    # 질량중심 찾기 함수, 차선 검출에서 사용됌
    def findCenterofMass(self, src):
        sum_of_y_mass_coordinates = 0
        num_of_mass_points = 0

        for y in range(0, len(src)):
            for x in range(0, len(src[0])):
                if src[y][x] == 255:
                    sum_of_y_mass_coordinates += y
                    num_of_mass_points += 1

        if num_of_mass_points == 0:
            center_of_mass_y = -1

        else:
            center_of_mass_y = int(round(sum_of_y_mass_coordinates / num_of_mass_points))

        return center_of_mass_y

    # 이미지 전처리 함수: 왜곡 보정, 시점 변환을 수행함
    def pretreatment(self, src, camera_matrix, distortion_matrix, transform_matrix, output_size):
        undistorted = cv2.undistort(src, camera_matrix, distortion_matrix, None, None)[10:438, 10:790]
        dst = cv2.warpPerspective(undistorted, transform_matrix, output_size)
        return dst

    def frm_pretreatment_parking(self, ret, frame, crop, *preParam):
        dst = self.pretreatment(frame, *preParam)
        cropped = dst[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        return cropped

    def frm_pretreatment(self, ret, frame, crop, *preParam):
        dst = self.pretreatment(frame, *preParam)
        cropped = dst[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        transposed = cv2.flip(cv2.transpose(cropped), 0)
        return transposed

    def make_filtered_frame(self):
        left_frame = self.frm_pretreatment(*self.video_left.read(), *LaneCam.xreadParam_L)
        right_frame = self.frm_pretreatment(*self.video_right.read(), *LaneCam.xreadParam_R)
        left_hsv = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)

        # HSV, RGB 필터링으로 영상을 이진화 함
        filtered_L = cv2.inRange(left_hsv, self.lower_yellow, self.upper_yellow)
        filtered_R = cv2.inRange(right_frame, self.lower_white, self.upper_white)

        filtered_both = np.vstack((filtered_R, filtered_L))
        final = cv2.flip(cv2.transpose(filtered_both), 1)

        self.filtered_both = final

    def default_loop(self):
        # 프레임 읽어들여서 왼쪽만 HSV 색공간으로 변환하기
        left_frame = self.frm_pretreatment(*self.video_left.read(), *LaneCam.xreadParam_L)
        right_frame = self.frm_pretreatment(*self.video_right.read(), *LaneCam.xreadParam_R)
        left_hsv = cv2.cvtColor(left_frame, cv2.COLOR_BGR2HSV)

        # HSV, RGB 필터링으로 영상을 이진화 함
        filtered_L = cv2.inRange(left_hsv, self.lower_yellow, self.upper_yellow)
        filtered_R = cv2.inRange(right_frame, self.lower_white, self.upper_white)

        # 좌, 우 영상을 붙임. 모니터링을 위한 부분.
        both = np.vstack((right_frame, left_frame))
        both_final = cv2.flip(cv2.transpose(both), 1)
        self.lane_cam_raw_frame.write(both_final)

        # ---------------------------------- 여기부터 왼쪽 차선 박스 쌓기 영역 ----------------------------------
        if self.left_previous_points is None:
            row_sum = np.sum(filtered_L[0:300, 200:300], axis=1)
            start_point = np.argmax(row_sum)

            # 차선의 실마리를 찾을 때, 길이가 7650 / 255 = 30픽셀 이상 될때만 차선으로 인정하고, 그렇지 않을 경우
            # 차선이 없는 것으로 간주함
            if (row_sum[start_point] > 7650):
                self.left_current_points = np.array([0] * 10)
                self.left_current_points[0] = start_point

                for i in range(1, 10):
                    reference = self.left_current_points[i - 1] - self.BOX_WIDTH

                    x1, x2 = 300 - 30 * i, 330 - 30 * i
                    y1 = self.left_current_points[i - 1] - self.BOX_WIDTH
                    y2 = self.left_current_points[i - 1] + self.BOX_WIDTH

                    small_box = filtered_L[y1:y2, x1:x2]
                    center_of_mass = self.findCenterofMass(small_box)

                    # 박스가 비어 있는 경우 -1을 저장
                    if center_of_mass == -1: self.left_current_points[i] = -1
                    else:
                        location = reference + center_of_mass
                        # 질량중심 결과가 전체 영상을 벗어나지 않았을 때만 저장하고
                        if 0 <= location < 300: self.left_current_points[i] = location
                        # 벗어나면 -1을 저장함
                        else: self.left_current_points[i] = -1

            else: self.left_current_points = None

        else:
            for i in range(0, 10):

                if self.left_current_points[i] != -1:
                    reference = self.left_previous_points[i] - self.BOX_WIDTH

                    x1, x2 = 270 - 30 * i, 300 - 30 * i
                    y1 = self.left_previous_points[i] - self.BOX_WIDTH
                    y2 = self.left_previous_points[i] + self.BOX_WIDTH

                    small_box = filtered_L[y1:y2, x1:x2]
                    center_of_mass = self.findCenterofMass(small_box)

                    if center_of_mass == -1: self.left_current_points[i] = -1
                    else:
                        location = reference + center_of_mass

                        if 0 <= location < 300: self.left_current_points[i] = location
                        else: self.left_current_points[i] = -1

                else:
                    if i == 0:
                        reference = self.left_previous_points[1] - self.BOX_WIDTH
                        x1, x2 = 270, 300
                        y1 = self.left_previous_points[1] - self.BOX_WIDTH
                        y2 = self.left_previous_points[1] + self.BOX_WIDTH

                        small_box = filtered_L[y1:y2, x1:x2]
                        center_of_mass = self.findCenterofMass(small_box)

                        if center_of_mass == -1: self.left_current_points[0] = -1
                        else:
                            location = reference + center_of_mass

                            if 0 <= location < 300:
                                self.left_current_points[0] = location
                            else:
                                self.left_current_points[0] = -1

                    else:
                        reference = self.left_previous_points[i - 1] - self.BOX_WIDTH

                        x1, x2 = 270 - 30 * i, 300 - 30 * i
                        y1 = self.left_previous_points[i - 1] - self.BOX_WIDTH
                        y2 = self.left_previous_points[i - 1] + self.BOX_WIDTH

                        small_box = filtered_L[y1:y2, x1:x2]
                        center_of_mass = self.findCenterofMass(small_box)

                        if center_of_mass == -1:
                            self.left_current_points[i] = -1
                        else:
                            location = reference + center_of_mass

                            if 0 <= location < 300:
                                self.left_current_points[i] = location
                            else:
                                self.left_current_points[i] = -1

        if np.count_nonzero(self.left_current_points == -1) >= 5: self.left_current_points = None
        self.left_previous_points = self.left_current_points
        # ---------------------------------- 여기까지 왼쪽 차선 박스 쌓기 영역 ----------------------------------

        # ---------------------------------- 여기부터 오른쪽 차선 박스 쌓기 영역 ----------------------------------
        if self.right_previous_points is None:
            row_sum = np.sum(filtered_R[0:300, 200:300], axis=1)
            start_point = np.argmax(row_sum)

            # 차선의 실마리를 찾을 때, 길이가 7650 / 255 = 30픽셀 이상 될때만 차선으로 인정하고, 그렇지 않을 경우
            # 차선이 없는 것으로 간주함
            if (row_sum[start_point] > 7650):
                self.right_current_points = np.array([0] * 10)
                self.right_current_points[0] = start_point

                for i in range(1, 10):
                    reference = self.right_current_points[i - 1] - self.BOX_WIDTH

                    x1, x2 = 300 - 30 * i, 330 - 30 * i
                    y1 = self.right_current_points[i - 1] - self.BOX_WIDTH
                    y2 = self.right_current_points[i - 1] + self.BOX_WIDTH

                    small_box = filtered_R[y1:y2, x1:x2]
                    center_of_mass = self.findCenterofMass(small_box)

                    # 박스가 비어 있는 경우 -1을 저장
                    if center_of_mass == -1: self.right_current_points[i] = -1
                    else:
                        location = reference + center_of_mass
                        # 질량중심 결과가 전체 영상을 벗어나지 않았을 때만 저장하고
                        if 0 <= location < 300: self.right_current_points[i] = location
                        # 벗어나면 -1을 저장함
                        else: self.right_current_points[i] = -1

            else: self.right_current_points = None

        else:
            for i in range(0, 10):

                if self.right_current_points[i] != -1:
                    reference = self.right_previous_points[i] - self.BOX_WIDTH

                    x1, x2 = 270 - 30 * i, 300 - 30 * i
                    y1 = self.right_previous_points[i] - self.BOX_WIDTH
                    y2 = self.right_previous_points[i] + self.BOX_WIDTH

                    small_box = filtered_R[y1:y2, x1:x2]
                    center_of_mass = self.findCenterofMass(small_box)

                    if center_of_mass == -1: self.right_current_points[i] = -1
                    else:
                        location = reference + center_of_mass

                        if 0 <= location < 300: self.right_current_points[i] = location
                        else: self.right_current_points[i] = -1

                else:
                    if i == 0:
                        reference = self.right_previous_points[1] - self.BOX_WIDTH
                        x1, x2 = 270, 300
                        y1 = self.right_previous_points[1] - self.BOX_WIDTH
                        y2 = self.right_previous_points[1] + self.BOX_WIDTH

                        small_box = filtered_L[y1:y2, x1:x2]
                        center_of_mass = self.findCenterofMass(small_box)

                        if center_of_mass == -1: self.right_current_points[0] = -1
                        else:
                            location = reference + center_of_mass

                            if 0 <= location < 300:
                                self.right_current_points[0] = location
                            else:
                                self.right_current_points[0] = -1

                    else:
                        reference = self.right_previous_points[i - 1] - self.BOX_WIDTH

                        x1, x2 = 270 - 30 * i, 300 - 30 * i
                        y1 = self.right_previous_points[i - 1] - self.BOX_WIDTH
                        y2 = self.right_previous_points[i - 1] + self.BOX_WIDTH

                        small_box = filtered_R[y1:y2, x1:x2]
                        center_of_mass = self.findCenterofMass(small_box)

                        if center_of_mass == -1:
                            self.right_current_points[i] = -1
                        else:
                            location = reference + center_of_mass

                            if 0 <= location < 300:
                                self.right_current_points[i] = location
                            else:
                                self.right_current_points[i] = -1

        if np.count_nonzero(self.right_current_points == -1) >= 5: self.right_current_points = None
        self.right_previous_points = self.right_current_points

        # ---------------------------------- 여기까지 오른쪽 차선 박스 쌓기 영역 ----------------------------------

        if self.left_current_points is not None:
            xs_valid = []
            ys_L_valid = []

            for i in range(0, 10):
                temp = self.left_current_points[i]
                if temp != -1:
                    xs_valid.append(-30 * i)
                    ys_L_valid.append(-1 * temp)
                    cv2.line(filtered_L, (300 - 30 * i, temp - self.BOX_WIDTH), (300 - 30 * i, temp + self.BOX_WIDTH), 150)

            self.left_coefficients = np.polyfit(xs_valid, ys_L_valid, 2)

            xs_plot = np.array([1 * i for i in range(-299, 1)])
            ys_plot_L = np.array(
                [self.left_coefficients[2] + self.left_coefficients[1] * v + self.left_coefficients[0] * v ** 2 for v in xs_plot])

            transformed_x = xs_plot + 299
            transformed_y_L = 0 - ys_plot_L

            for i in range(0, 300):
                cv2.circle(filtered_L, (int(transformed_x[i]), int(transformed_y_L[i])), 2, 150, -1)

        else: self.left_coefficients = None

        if self.right_current_points is not None:
            xs_valid = []
            ys_R_valid = []

            for i in range(0, 10):
                temp = self.right_current_points[i]
                if temp != -1:
                    xs_valid.append(-30 * i)
                    ys_R_valid.append(300 - temp)
                    cv2.line(filtered_R, (300 - 30 * i, temp - self.BOX_WIDTH), (300 - 30 * i, temp + self.BOX_WIDTH), 150)

            self.right_coefficients = np.polyfit(xs_valid, ys_R_valid, 2)

            xs_plot = np.array([1 * i for i in range(-299, 1)])
            ys_plot_R = np.array(
                [self.right_coefficients[2] + self.right_coefficients[1] * v + self.right_coefficients[0] * v ** 2 for v in xs_plot])

            transformed_x = xs_plot + 299
            transformed_y_R = 299 - ys_plot_R

            for i in range(0, 300):
                cv2.circle(filtered_R, (int(transformed_x[i]), int(transformed_y_R[i])), 2, 150, -1)

        else: self.right_coefficients = None

        filtered_both = np.vstack((filtered_R, filtered_L))
        final = cv2.flip(cv2.transpose(filtered_both), 1)

        self.lane_cam_frame.write(final)
 
    def stopline_loop(self):
        pass

    def parkingline_loop(self):
        parking_frame = self.frm_pretreatment_parking(*self.video_right.read(), *LaneCam.xreadparam_R_parking)
        filtered_R = cv2.inRange(parking_frame, self.lower_white, self.upper_white)

        lines = cv2.HoughLinesP(filtered_R, 1, np.pi / 360, 40, 10, 10)
        t1 = time.time()

        while True:
            if lines is None or len(lines) <= 1:
                v1 = None
                v2 = None
                break

            rand = random.sample(range(0, len(lines)), 2)

            for x1, y1, x2, y2 in lines[rand[0]]: v1 = (x1 - x2, y2 - y1)
            for x1, y1, x2, y2 in lines[rand[1]]: v2 = (x1 - x2, y2 - y1)

            magnitude_1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            magnitude_2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]

            cos_theta = dot_product / (magnitude_1 * magnitude_2)
            if cos_theta > 1: cos_theta = 1
            theta = np.rad2deg(np.arccos(cos_theta))

            if (50 < theta < 70 or 110 < theta < 130) and min(magnitude_1, magnitude_2) > 40:
                break

            if (time.time() - t1) >= 0.01:
                v1 = None
                v2 = None
                break

        if v1 is not None and v2 is not None:
            theta1 = np.arctan2(v1[1], v1[0])
            theta2 = np.arctan2(v2[1], v2[0])

            if theta1 <= -np.deg2rad(90): theta1 += np.deg2rad(180)
            elif theta1 > np.deg2rad(120): theta1 -= np.deg2rad(180)

            if theta2 <= -np.deg2rad(90): theta2 += np.deg2rad(180)
            elif theta2 > np.deg2rad(120): theta2 -= np.deg2rad(180)

            middle = (theta1 + theta2) / 2

            lane_points = lines[rand[0]]
            parkline_points = lines[rand[1]]

            A = np.array([[lane_points[0][2] - lane_points[0][0], parkline_points[0][0] - parkline_points[0][2]],
                          [lane_points[0][1] - lane_points[0][3], parkline_points[0][3] - parkline_points[0][1]]])

            b = np.array([parkline_points[0][0] - lane_points[0][0], lane_points[0][1] - parkline_points[0][1]])

            x, y = None, None

            try:
                solution = np.linalg.solve(A, b)

                x = int(lane_points[0][0] + solution[0] * (lane_points[0][2] - lane_points[0][0]))
                y = 200 - int(200 - lane_points[0][1] + solution[0] * (lane_points[0][1] - lane_points[0][3]))

                probe_start_x, probe_start_y = int(x + 300 * np.cos(middle)), int(y - 300 * np.sin(middle))

                cv2.circle(parking_frame, (x, y), 7, (0, 255, 0), -1)
                cv2.line(parking_frame, (x, y), (probe_start_x, probe_start_y), (255, 0, 0), 2)

            except:
                pass

            self.parkingline_info = (x, 200 - y, middle, min(theta1, theta2), max(theta1, theta2))

        else:
            self.parkingline_info = None

        # parking_frame을 모니터에 넘겨줘야 함.
        self.parkingline_frame.write(parking_frame)

    def stop(self):
        self.video_left.release()
        self.video_right.release()


if __name__ == "__main__":
    from monitor import Monitor
    monitor = Monitor()
    lane_cam = LaneCam()
    while True:
        lane_cam.parkingline_loop()
        monitor.show('1', *lane_cam.getFrame())
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    lane_cam.stop()