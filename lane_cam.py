import cv2
import numpy as np
import threading
import time
np.set_printoptions(linewidth=100000)


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

    # BGR 을 이용한 차선 추출에 필요한 값들
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 180, 230])

    lower_grey = np.array([150, 150, 150])
    upper_grey = np.array([210, 210, 210])

    BOX_WIDTH = 30

    def __init__(self):
        # 웹캠 2대 열기
        self.video_left = cv2.VideoCapture('output_L_0.avi')
        self.video_right = cv2.VideoCapture('output_R_0.avi')

        # 양쪽 웹캠의 해상도를 800x448로 설정
        '''
        self.video_left.set(3, 800)
        self.video_left.set(4, 448)
        self.video_right.set(3, 800)
        self.video_right.set(4, 448)
        '''

        # 현재 읽어온 프레임이 실시간으로 업데이트됌
        self.left_frame = None
        self.right_frame = None

        # 이전 프레임의 ROI 위치를 저장함
        self.left_previous_points = None
        self.right_previous_points = None

        # 현재 프레임의 ROI 위치를 저장함
        self.left_current_points = np.array([0] * 10)
        self.right_current_points = np.array([0] * 10)

        # 차선과 닮은 이차함수의 계수 세 개를 담음
        self.left_coefficients = None
        self.right_coefficients = None

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
            center_of_mass_y = int(round(len(src) / 2))

        else:
            center_of_mass_y = int(round(sum_of_y_mass_coordinates / num_of_mass_points))

        return center_of_mass_y

    # 이미지 전처리 함수: 왜곡 보정, 시점 변환을 수행함
    def pretreatment(self, src, camera_matrix, distortion_matrix, transform_matrix, output_size):

        undistorted = cv2.undistort(src, camera_matrix, distortion_matrix, None, None)[10:438, 10:790]
        dst = cv2.warpPerspective(undistorted, transform_matrix, output_size)

        return dst

    def left_camera_loop(self):
        while True:
            ret_L, frame_L = self.video_left.read()

            dst_L = self.pretreatment(frame_L, self.camera_matrix_L,
                                    self.distortion_coefficients_L, self.Bird_view_matrix_L, (563, 511))
            cropped_L = dst_L[210:510, 262:562]
            transposed_L = cv2.flip(cv2.transpose(cropped_L), 0)

            self.left_frame = transposed_L

    def right_camera_loop(self):
        while True:
            ret_R, frame_R = self.video_right.read()

            dst_R = self.pretreatment(frame_R, self.camera_matrix_R,
                                      self.distortion_coefficients_R, self.Bird_view_matrix_R, (503, 452))

            cropped_R = dst_R[151:451, 0:300]

            transposed_R = cv2.flip(cv2.transpose(cropped_R), 0)

            self.right_frame = transposed_R

    def show_loop(self):
        time.sleep(3)
        while True:
            left_frame, right_frame = self.left_frame, self.right_frame
            both = np.vstack((right_frame, left_frame))

            black_filtered_L = cv2.inRange(left_frame, self.lower_black, self.upper_black)
            black_filtered_R = cv2.inRange(right_frame, self.lower_black, self.upper_black)

            grey_filtered_L = cv2.inRange(left_frame, self.lower_grey, self.upper_grey)
            grey_filtered_R = cv2.inRange(right_frame, self.lower_grey, self.upper_grey)

            filtered_L = cv2.bitwise_not(cv2.bitwise_or(black_filtered_L, grey_filtered_L))
            filtered_R = cv2.bitwise_not(cv2.bitwise_or(black_filtered_R, grey_filtered_R))

            both_filtered = np.vstack((filtered_R, filtered_L))
            cv2.imshow('1', cv2.flip(cv2.transpose(both_filtered), 1))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            if self.left_previous_points is None:
                row_sum = np.sum(filtered_L, axis=1)
                start_point = np.argmax(row_sum)
                self.left_current_points[0] = start_point

                for i in range(1, 10):
                    reference = self.left_current_points[i - 1] - self.BOX_WIDTH

                    x1, x2 = 300 - 30 * i, 330 - 30 * i
                    y1, y2 = self.left_current_points[i - 1] - self.BOX_WIDTH, self.left_current_points[i - 1] + self.BOX_WIDTH

                    small_box = filtered_L[y1:y2, x1:x2]

                    self.left_current_points[i] = reference + self.findCenterofMass(small_box)

            else:
                for i in range(0, 10):
                    reference = self.left_previous_points[i] - self.BOX_WIDTH

                    x1, x2 = 270 - 30 * i, 300 - 30 * i
                    y1, y2 = self.left_previous_points[i] - self.BOX_WIDTH, self.left_previous_points[i] + self.BOX_WIDTH

                    small_box = filtered_L[y1:y2, x1:x2]

                    self.left_current_points[i] = reference + self.findCenterofMass(small_box)

            self.left_previous_points = self.left_current_points

            if self.right_previous_points is None:
                row_sum = np.sum(filtered_R, axis=1)
                start_point = np.argmax(row_sum)
                self.right_current_points[0] = start_point

                for i in range(1, 10):
                    reference = self.right_current_points[i - 1] - self.BOX_WIDTH

                    x1, x2 = 300 - 30 * i, 330 - 30 * i
                    y1, y2 = self.right_current_points[i - 1] - self.BOX_WIDTH, self.right_current_points[i - 1] + self.BOX_WIDTH

                    small_box = filtered_R[y1:y2, x1:x2]

                    self.right_current_points[i] = reference + self.findCenterofMass(small_box)

            else:
                for i in range(0, 10):
                    reference = self.right_previous_points[i] - self.BOX_WIDTH

                    x1, x2 = 270 - 30 * i, 300 - 30 * i
                    y1, y2 = self.right_previous_points[i] - self.BOX_WIDTH, self.right_previous_points[i] + self.BOX_WIDTH

                    small_box = filtered_R[y1:y2, x1:x2]

                    self.right_current_points[i] = reference + self.findCenterofMass(small_box)

            self.right_previous_points = self.right_current_points

            for i in range(0, 10):
                cv2.line(filtered_L, (300 - 30 * i, self.left_current_points[i] - self.BOX_WIDTH), (300 - 30 * i, self.left_current_points[i] + self.BOX_WIDTH), 150)
                cv2.line(filtered_R, (300 - 30 * i, self.right_current_points[i] - self.BOX_WIDTH), (300 - 30 * i, self.right_current_points[i] + self.BOX_WIDTH), 150)

            xs = np.array([30 * i for i in range(10, 0, -1)]) - 300
            ys_L = 0 - self.left_current_points
            ys_R = 300 - self.right_current_points

            coefficients_L = np.polyfit(xs, ys_L, 2)
            coefficients_R = np.polyfit(xs, ys_R, 2)

            self.left_coefficients = coefficients_L
            self.right_coefficients = coefficients_R

            xs_plot = np.array([1 * i for i in range(-299, 1)])
            ys_plot_L = np.array([coefficients_L[2] + coefficients_L[1] * v + coefficients_L[0] * v ** 2 for v in xs_plot])
            ys_plot_R = np.array([coefficients_R[2] + coefficients_R[1] * v + coefficients_R[0] * v ** 2 for v in xs_plot])

            transformed_x = xs_plot + 299
            transformed_y_L =  0 - ys_plot_L
            transformed_y_R = 299 - ys_plot_R

            for i in range(0, 300):
                cv2.circle(filtered_L, (int(transformed_x[i]), int(transformed_y_L[i])), 2, 150, -1)
                cv2.circle(filtered_R, (int(transformed_x[i]), int(transformed_y_R[i])), 2, 150, -1)

            cv2.imshow('left', filtered_L)
            cv2.imshow('right', filtered_R)
            cv2.imshow('2', cv2.flip(cv2.transpose(both), 1))
            #time.sleep(0.2)
            if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == "__main__":
    lane_cam = LaneCam()

    t1 = threading.Thread(target=lane_cam.left_camera_loop)
    t2 = threading.Thread(target=lane_cam.right_camera_loop)
    t3 = threading.Thread(target=lane_cam.show_loop)

    t1.start()
    t2.start()
    t3.start()