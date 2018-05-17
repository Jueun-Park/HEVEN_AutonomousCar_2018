# 2018-05-10 현지웅
# 현재 표지판 상황.
# 영상 데이터를 받아서 매 프레임마다 shape detect 후의 사진을 받아서 학습한 CNN 모델로 어느 표지판인지 인식하고 그에 대한 확률을 보여주고 가장 높은 확률을 return 하여, count를 할 수 있게 해놓음.
#
# 하지만 문제점
# 1. 표지판이 아닌 것을 shape deect 후 넘겨주는 경우가 빈번.
# -> 그런데 학습한 모델이 그런 경우를 95%이상의 확률로 어느 표지판이라고 인식하는 경우가 있음.
# 2. 한 프레임별로 하지 말고 띄엄띄엄해야할 것 같은데 아직 어떻게 하는지 몰라서 못하는 중.

from matplotlib import pyplot as plt
import sys
import numpy as np
import os
import tensorflow as tf

import cv2
import time
from shape_detection import shape_detect

import threading

'''
우선 구현 방법은 Tensorflow/model의 slim이라는 tensorflow가 제공하는 틀을 이용할 거임
이 방법을 사용한 이유는 실제로 데이터를 구현하고 tensorflow로 표지판을 인식하는데까지 너무 많은 노력과 지식이 필요한데
그것을 충당할 수 있는 시간이 없어서 tensorflow에서 제공하는 모듈을 사용하기로 함
'''
# sys.path.insert(0, './slim')
# 이 부분이 중요! 아래에 nets와 preprocessing은 tensorflow/model안에 slim이라는 폴더 안에 있는 폴더로써 slim파일을 불러와야 작동이 됨
# 따라서 만약 Tensorflow/model파일이 없으면 https://github.com/tensorflow/models/ 여기에 들어가서 다운받은후에 slim 디렉토리를 위에 넣어줌

# from nets import inception
# from preprocessing import inception_preprocessing


class SignCam:
    def __init__(self):
        self.sign_trigger = 0
        self.is_in_mission = False
        self.sign = [[0 for col in range(7)] for row in range(2)]
        self.cam = cv2.VideoCapture(0)  # r'C:\Users\Administrator\PycharmProjects\Lane_logging\cut.mp4') #2
        self.cam.set(3, 800)
        self.cam.set(4, 448)
        self.sign2action = "Nothing"
        self.mission_number = 0


        self.thread = threading.Thread(target=self.detect_one_frame)
        self.stop_fg = False
        self.exit_fg = False


        self.sign_init()

    # modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
    #           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

    def sign_control(self):
        return self.sign_trigger

    def start(self):
        self.thread.start()

    def restart(self):
        self.stop_fg = False

    def stop(self):
        self.stop_fg = True

    def exit(self):
        self.exit_fg = True

    def sign_init(self):
        self.sign[0][0] = 'Bicycles'  # MOVING_OBS 3
        self.sign[0][1] = 'Crosswalk_PedestrainCrossing'  # CROSS_WALK 7
        self.sign[0][2] = 'Double_bend'  # S_Curve 4
        self.sign[0][3] = 'Narrow_Carriageway'  # NARROW 5
        self.sign[0][4] = 'Parking_Lot'  # PARKING 1
        self.sign[0][5] = 'Roadworks'  # STATIC_OBS 2
        self.sign[0][6] = 'u_turn'  # U_TURN 6
        self.sign[1][0] = 0
        self.sign[1][1] = 0
        self.sign[1][2] = 0
        self.sign[1][3] = 0
        self.sign[1][4] = 0
        self.sign[1][5] = 0
        self.sign[1][6] = 0

    def sign_reinit(self):
        self.sign[1][0] = 0
        self.sign[1][1] = 0
        self.sign[1][2] = 0
        self.sign[1][3] = 0
        self.sign[1][4] = 0
        self.sign[1][5] = 0
        self.sign[1][6] = 0

    def countup_recognition(self, result_sign, prob):
        for i in range(7):
            if self.sign[0][i] == result_sign and prob > 0.95:
                self.sign[1][i] = self.sign[1][i] + 1
                break

    def print_sign(self):  # test code 에서 사용될 출력 함수
        for i in range(7):
            print("print_sign: ", self.sign[0][i], self.sign[1][i])

    def set_sign2action(self):
        # 만약 한 표지판의 인식 횟수가 1회 이상이 되면, 그 sign에 대한 action을 준비하고, 횟수 모두 초기화하기
        for i in range(7):
            if self.sign[1][i] >= 1:
                self.sign2action = self.sign[0][i]
                self.sign[1][0] = 0
                self.sign[1][1] = 0
                self.sign[1][2] = 0
                self.sign[1][3] = 0
                self.sign[1][4] = 0
                self.sign[1][5] = 0
                self.sign[1][6] = 0
                break

    def detect_one_frame(self):
        while True:
            if self.exit_fg is True: break
            if self.stop_fg is True: time.sleep(1); continue
            # while (self.cam.isOpened()):
            frame_okay, frame = self.cam.read()  # 한 프레임을 가져오자.
            # 이미지 중 표지판이 있는 곳 확인




            img_list = shape_detect(frame)

            if len(img_list) == 0:
                self.sign_trigger = 0
            else:
                self.sign_trigger = 1


            cv2.imshow('1', frame)

            if cv2.waitKey(1) & 0xff == 27:
                return
            for img in img_list:
                result_sign, prob = self.process_one_frame_sign(img)
                print("result sign: ", result_sign)
                self.countup_recognition(result_sign, prob)

            # self.print_sign()
            self.set_sign2action()



    def get_mission(self):
        # modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
        #           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}
        if self.sign2action == "Nothing":
            self.mission_number = 0
        elif self.sign2action == 'Parking_Lot':
            self.mission_number = 1
        elif self.sign2action == 'Roadworks':
            self.mission_number = 2
        elif self.sign2action == 'Bicycles':
            self.mission_number = 3
        elif self.sign2action == 'Double_bend':
            self.mission_number = 4
        elif self.sign2action == 'Narrow_Carriageway':
            self.mission_number = 5
        elif self.sign2action == 'u_turn':
            self.mission_number = 6
        elif self.sign2action == 'Crosswalk_PedestrainCrossing':
            self.mission_number = 7

        if self.mission_number > 0:
            self.sign_reinit()

        self.sign2action = "Nothing"
        # print(self.mission_number)
        return self.mission_number

    def is_in_this_mission(self, ndarray):
        try:
            if np.sum(ndarray) >= 1:
                return True
            else:
                return False
        except Exception as e:
            return False


    def process_one_frame_sign(self, frame):
        if len(frame) < 1:
            return "Nothing", 0.00

        # 프레임 시작 시간 측정
        t1 = time.time()

        cv2.imwrite('test.jpg',frame)
        image_data = tf.gfile.FastGFile('test.jpg','rb').read()

        #label_lines = [line.rstrip() for line in tf.gfile.GFile('')]
        label_lines = ['Bicycles', 'Crosswalk_PedestrainCrossing', 'Double_bend', 'Narrow_Carriageway', 'Parking_Lot',
                 'Roadworks', 'u_turn']

        with tf.gfile.FastGFile(
                "C:/Users/Administrator/Desktop/HEVEN_AutonomousCar_2018/deep_learning/minimal_graph.proto", 'rb') as f:
            # with tf.device('/gpu:0'):
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        # config=tf.ConfigProto(log_device_placement=True)

        with tf.Session() as sess:
            # Feed the image_data as input to the graph and get first prediction
            # Feed data tensor 이름 각각 입력 (softmax_tensor가 y_eval, predictions가 x 데이터인듯)
            softmax_tensor = sess.graph.get_tensor_by_name('InceptionV1/Logits/Predictions/Softmax:0')
            predictions = sess.run(softmax_tensor, {'input_image:0': image_data})
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            print(top_k)

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print()
                print('%s (score = %.5f)' % (human_string, score))


        t2 = time.time()

        print("one frame time: ", t2 - t1)

        # 가장 높은 확률인 표지판 이름과 확률을 return해줌으로서 count를 할 수 있도록 함.
        return label_lines[top_k[0]], predictions[0][top_k[0]]


if __name__ == "__main__":
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    current_signcam = SignCam()
    current_signcam.start()

    while (current_signcam.cam.isOpened()):
        #current_signcam.detect_one_frame()
        mission_number = current_signcam.get_mission()
        #print(mission_number)
