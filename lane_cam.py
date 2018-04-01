import cv2
import numpy as np
import threading
import time


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
    upper_black = np.array([180, 180, 180])

    lower_grey = np.array([150, 150, 150])
    upper_grey = np.array([210, 210, 210])

    def __init__(self):
        # 웹캠 2대 열기
        self.video_left = cv2.VideoCapture(1)
        self.video_right = cv2.VideoCapture(0)

        # 양쪽 웹캠의 해상도를 800x448로 설정
        self.video_left.set(3, 800)
        self.video_left.set(4, 448)
        self.video_right.set(3, 800)
        self.video_right.set(4, 448)

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
        self.coefficients = None

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
            ret, frame = self.video_left.read()
            dst = self.pretreatment(frame, self.camera_matrix_L,
                                    self.distortion_coefficients_L, self.Bird_view_matrix_L, (563, 511))

            cropped = dst[210:510, 262:562]

            #transposed = cv2.flip(cv2.transpose(dst), 0)
            self.left_frame = cropped

            '''
            black_filtered = cv2.inRange(transposed, self.lower_black, self.upper_black)
            grey_filtered = cv2.inRange(transposed, self.lower_grey, self.upper_grey)
            filtered = cv2.bitwise_not(black_filtered + grey_filtered)

            if self.left_previous_points is None:
                row_sum = np.sum(filtered[0:300, 270:300], axis=1)[100:200]
                start_point = np.argmax(row_sum) + 100
                self.left_current_points[0] = start_point

                for i in range(1, 10):
                    reference = self.left_current_points[i - 1] - 20

                    x1, x2 = 300 - 30 * i, 330 - 30 * i
                    y1, y2 = self.left_current_points[i - 1] - 20, self.left_current_points[i - 1] + 20

                    small_box = filtered[y1:y2, x1:x2]

                    self.left_current_points[i] = reference + self.findCenterofMass(small_box)

            else:
                for i in range(0, 10):
                    reference_ = self.left_previous_points[i] - 20

                    x1_, x2_ = 270 - 30* i, 300 - 30 * i
                    y1_, y2_ = self.left_previous_points[i] - 20, self.left_previous_points[i] + 20

                    small_box_ = filtered[y1_:y2_, x1_:x2_]

                    self.left_current_points[i] = reference_ + self.findCenterofMass(small_box_)

            self.left_previous_points = self.left_current_points'''


    def right_camera_loop(self):

        while True:

            ret, frame = self.video_right.read()
            dst = self.pretreatment(frame, self.camera_matrix_R,
                                    self.distortion_coefficients_R, self.Bird_view_matrix_R, (503, 452))

            #transposed = cv2.flip(cv2.transpose(dst), 0)
            cropped = dst[151:451, 0:300]

            self.right_frame = cropped


            '''
            black_filtered = cv2.inRange(transposed, self.lower_black, self.upper_black)
            grey_filtered = cv2.inRange(transposed, self.lower_grey, self.upper_grey)
            filtered = cv2.bitwise_not(black_filtered + grey_filtered)

            if self.right_previous_points is None:
                row_sum = np.sum(filtered[0:300, 270:300], axis=1)[100:200]
                start_point = np.argmax(row_sum) + 100
                self.right_current_points[0] = start_point

                for i in range(1, 10):
                    reference = self.right_current_points[i - 1] - 20

                    x1, x2 = 300 - 30 * i, 330 - 30 * i
                    y1, y2 = self.right_current_points[i - 1] - 20, self.right_current_points[i - 1] + 20

                    small_box = filtered[y1:y2, x1:x2]

                    self.right_current_points[i] = reference + self.findCenterofMass(small_box)

            else:
                for i in range(0, 10):
                    reference_ = self.right_previous_points[i] - 20

                    x1_, x2_ = 270 - 30* i, 300 - 30 * i
                    y1_, y2_ = self.right_previous_points[i] - 20, self.right_previous_points[i] + 20

                    small_box_ = filtered[y1_:y2_, x1_:x2_]

                    self.right_current_points[i] = reference_ + self.findCenterofMass(small_box_)

            self.left_previous_points = self.left_current_points'''

    def show_loop(self):
        while True:
            if self.left_frame is not None and self.right_frame is not None:
                both = np.hstack((self.left_frame, self.right_frame))
                cv2.imshow('both', both)

            else: print("카메라가 연결되지 않았거나, 아직 준비중입니다.")

            if cv2.waitKey(1) & 0xFF == ord('q'): break

    def initiateleft(self):
        left_thread = threading.Thread(target=self.left_camera_loop)
        left_thread.start()
    def initiateright(self):
        right_thread = threading.Thread(target=self.right_camera_loop)
        right_thread.start()
    def initiateshow(self):
        showing_thread = threading.Thread(target=self.show_loop)
        showing_thread.start()


'''
    xs = np.array([30 * i for i in range(10, 0, -1)]) - 300
    ys = 300 - current_points

    coefficients = np.polyfit(xs, ys, 2)
    print(coefficients)

    xs_plot = np.array([1 * i for i in range(301, 0, -1)]) - 300
    ys_plot = np.array([coefficients[2] + coefficients[1] * v + coefficients[0] * v**2 for v in xs_plot])

    transformed_x = xs_plot + 300
    transformed_y = 300 - ys_plot

    for i in range(0, 301):
        cv2.circle(right_transposed, (int(transformed_x[i]), int(transformed_y[i])), 2, (0, 0, 200), -1)

    #for i in range(0, 10):
        #cv2.line(right_filtered, (300 - 30 * i, current_points[i] - 20), (300 - 30 * i, current_points[i] + 20), 140)

    cv2.imshow('right', right_transposed)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

video_left.release()
video_right.release()
cv2.destroyAllWindows()

lanecam = LaneCam()
lanecam.initiate()
'''
if __name__=="__main__" :
    lanecam=LaneCam()

    t1=threading.Thread(target=lanecam.initiateleft)
    t2 = threading.Thread(target=lanecam.initiateright)
    t3 = threading.Thread(target=lanecam.initiateshow)

    t1.start()
    t2.start()
    t3.start()



