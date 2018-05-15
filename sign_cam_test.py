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

'''
우선 구현 방법은 Tensorflow/model의 slim이라는 tensorflow가 제공하는 틀을 이용할 거임
이 방법을 사용한 이유는 실제로 데이터를 구현하고 tensorflow로 표지판을 인식하는데까지 너무 많은 노력과 지식이 필요한데
그것을 충당할 수 있는 시간이 없어서 tensorflow에서 제공하는 모듈을 사용하기로 함
'''
sys.path.insert(0, './slim')
#이 부분이 중요! 아래에 nets와 preprocessing은 tensorflow/model안에 slim이라는 폴더 안에 있는 폴더로써 slim파일을 불러와야 작동이 됨
#따라서 만약 Tensorflow/model파일이 없으면 https://github.com/tensorflow/models/ 여기에 들어가서 다운받은후에 slim 디렉토리를 위에 넣어줌

from nets import inception
from preprocessing import inception_preprocessing


def is_in_this_mission(ndarray):
    try:
        if np.sum(ndarray) >= 1:
            return True
        else:
            return False
    except Exception as e:
        return False

def process_one_frame_sign(frame, is_in_mission):
    if len(frame) < 1:
        return "Nothing", 0.00


    if is_in_mission:
        pass

    t1 = time.time()  # 프레임 시작 시간 측정

    checkpoints_dir = 'C:/Users/Administrator/Desktop/tmp/train_inception_v1_smartcar_logs'
    # 데이터의 checkpoint 디렉토리 넣어줌
    slim = tf.contrib.slim

    image_size = inception.inception_v1.default_image_size
    # 사용되는 딥러닝 툴은 inception v1으로 가동됨

    user_images = []
    user_processed_images = []

    cv2.imwrite('test.jpg',frame)
    image_input = tf.read_file('test.jpg')
    image = tf.image.decode_jpeg(image_input, channels=3)
    user_images.append(image)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    user_processed_images.append(processed_image)

    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(user_processed_images, num_classes=7, is_training=False, reuse=tf.AUTO_REUSE)
        # Number of class: 우리 표지판 총 7개의 class를 판별
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'model.ckpt-11542'),
        # Checkpoint 디렉토리에서 실제로 사용되는 최신 데이터
        slim.get_model_variables('InceptionV1'))
    # Slim model중 InceptionV1을 이용함

    with tf.Session() as sess:
        init_fn(sess)
        np_images, probabilities = sess.run([user_images, probabilities])

    #names = os.listdir("C:/Users/Administrator/Desktop/dataset/smartcar/smartcar_photos")
    # 7개 class의 이름을 불러오는 작업, smartcar_photos안에 총 7개의 표지판 이름으로 된 폴더가 있는데 그 이름들을 인식함
    names = ['Bicycles', 'Crosswalk_PedestrainCrossing', 'Double_bend', 'Narrow_Carriageway', 'Parking_Lot', 'Roadworks', 'u_turn']


    probabilitie = probabilities[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilitie), key=lambda x: x[1])]

    for p in range(7):
        index = sorted_inds[p]
        print('Probability %0.2f%% => [%s]' % (probabilitie[index], names[index]))

    # 검출된 표지판 사진 확인하고 싶으면 주석을 푸시면 됩니다. 근데 그 사진을 꺼야지 다음 코드가 진행이 되더라고요.
    # plt.figure()
    # plt.imshow(np_images[0].astype(np.uint8))
    # plt.axis('off')
    # plt.show()


    return names[sorted_inds[0]], probabilitie[sorted_inds[0]] # 가장 높은 확률인 표지판 이름과 확률을 return해줌으로서 count를 할 수 있도록 함.

class SignCam:
    def __init__(self):
        self.is_in_mission = False
        self.sign = [[0 for col in range(7)]for row in range(2)]
        self.cam = cv2.VideoCapture(2)
        self.sign2action = "Nothing"
        self.mission_number = 0

        self.sign_init()

    # modes = {'DEFAULT': 0, 'PARKING': 1, 'STATIC_OBS': 2,  'MOVING_OBS': 3,
    #           'S_CURVE': 4, 'NARROW': 5, 'U_TURN': 6, 'CROSS_WALK': 7}

    def sign_init(self):
        self.sign[0][0] = 'Bicycles' #
        self.sign[0][1] = 'Crosswalk_PedestrainCrossing' # CROSS_WALK 7
        self.sign[0][2] = 'Double_bend' # S_Curve 4
        self.sign[0][3] = 'Narrow_Carriageway' # NARROW 5
        self.sign[0][4] = 'Parking_Lot' # PARKING 1
        self.sign[0][5] = 'Roadworks' #
        self.sign[0][6] = 'u_turn'
        self.sign[1][0] = 0
        self.sign[1][1] = 0
        self.sign[1][2] = 0
        self.sign[1][3] = 0
        self.sign[1][4] = 0
        self.sign[1][5] = 0
        self.sign[1][6] = 0
        return self.sign

    def countup_recognition(self, result_sign, prob):
        for i in range(7):
            if self.sign[0][i] == result_sign and prob > 0.95:
                self.sign[1][i] = self.sign[1][i] + 1
                break
        return self.sign

    def print_sign(self): # test code 에서 사용될 출력 함수
        for i in range(7):
            print(self.sign[0][i], self.sign[1][i])

    def set_sign2action(self):
        for i in range(7):  # 만약 한 표지판의 인식 횟수가 1회 이상이 되면, 그 sign에 대한 action을 준비하고, 횟수 모두 초기화하기
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
        #while (self.cam.isOpened()):
            frame_okay, frame = self.cam.read()  # 한 프레임을 가져오자.
            # 이미지 중 표지판이 있는 곳 확인

            img_list = shape_detect(frame)

            print(img_list)
            for img in img_list:
                result_sign, prob = process_one_frame_sign(img, self.is_in_mission)
                print(result_sign)
                self.sign = self.countup_recognition(result_sign, prob)

            self.print_sign()
            self.set_sign2action()

    def get_mission(self):
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

        self.sign2action = "Nothing"
        print(self.mission_number)
        return self.mission_number


if __name__ == "__main__":
    current_signcam = SignCam()
    while(current_signcam.cam.isOpened()):
        start=time.time()
        current_signcam.detect_one_frame()
        mission_number = current_signcam.get_mission()
        print(mission_number)

