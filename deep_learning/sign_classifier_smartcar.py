# 2018-05-10 현지웅

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

sys.path.insert(0, 'C:/Users/Administrator/Desktop/slim')
#이 부분이 중요! 아래에 nets와 preprocessing은 tensorflow/model안에 slim이라는 폴더 안에 있는 폴더로써 slim파일을 불러와야 작동이 됨
#따라서 만약 Tensorflow/model파일이 없으면 https://github.com/tensorflow/models/ 여기에 들어가서 다운받은후에 slim 디렉토리를 위에 넣어줌

from nets import inception
from preprocessing import inception_preprocessing


checkpoints_dir = 'C:/Users/Administrator/Desktop/dataset/train_inception_v1_smartcar_FineTune_logs/all'
#데이터의 checkpoint 디렉토리 넣어줌
slim = tf.contrib.slim

image_size = inception.inception_v1.default_image_size
#사용되는 딥러닝 툴은 inception v1으로 가동됨


def process_one_frame_sign(frame, is_in_mission):
    if is_in_mission:
        pass

    t1 = time.time()  # 프레임 시작 시간 측정

    user_images = []
    user_processed_images = []

    image = tf.image.decode_jpeg(frame, channels=3)
    user_images.append(image)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    user_processed_images.append(processed_image)
    processed_images = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(user_processed_images, num_classes=7, is_training=False)
        # Number of class: 우리 표지판 총 7개의 class를 판별
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'model.ckpt-500'),
        # Checkpoint 디렉토리에서 실제로 사용되는 최신 데이터
        slim.get_model_variables('InceptionV1'))
    # Slim model중 InceptionV1을 이용함

    with tf.Session() as sess:
        init_fn(sess)
        np_images, probabilities = sess.run([user_images, probabilities])

    #names = os.listdir("C:/Users/Administrator/Desktop/dataset/smartcar/smartcar_photos")
    # 7개 class의 이름을 불러오는 작업, smartcar_photos안에 총 7개의 표지판 이름으로 된 폴더가 있는데 그 이름들을 인식함
    names = ['Bicycles', 'Crosswalk_PedestrainCrossing', 'Double_bend', 'Narrow_Carriageway', 'Parking_Lot', 'Roadworks', 'u_turn']


    probabilitie = probabilities[frame, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probabilitie), key=lambda x: x[1])]

    plt.figure()
    plt.imshow(np_images[frame].astype(np.uint8))
    plt.axis('off')
    plt.show()

    for p in range(7):
        index = sorted_inds[p]
        print('Probability %0.2f%% => [%s]' % (probabilitie[index], names[index]))


    return names[0] # 가장 높은 확률인 표지판을 return해줌으로서 count를 할 수 있도록 함.



if __name__ == "__main__":

    # 웹캠 읽어오기
    cam = cv2.VideoCapture(0)
    time.sleep(2)

    is_in_mission = False
    sign = ['Bicycles', 'Crosswalk_PedestrainCrossing', 'Double_bend', 'Narrow_Carriageway', 'Parking_Lot', 'Roadworks', 'u_turn']
    sign = [[]*7 for i in range(2)]
    sign[0][0] = 'Bicycles'
    sign[0][1] = 'Crosswalk_PedestrainCrossing'
    sign[0][2] = 'Double_bend'
    sign[0][3] = 'Narrow_Carriageway'
    sign[0][4] = 'Parking_Lot'
    sign[0][5] = 'Roadworks'
    sign[0][6] = 'u_turn'
    sign[1][0] = 0
    sign[1][1] = 0
    sign[1][2] = 0
    sign[1][3] = 0
    sign[1][4] = 0
    sign[1][5] = 0
    sign[1][6] = 0

    # 영상 처리 -> 이쪽에서 5m부터 1m 씩 조정할 수 있도록 짜야할 것 같음.
    while (True):
        frame_okay, frame = cam.read()  # 한 프레임을 가져오자.
        # 이미지 중 표지판이 있는 곳 확인
        img_list = shape_detect(frame)
        for img in img_list:
            result = process_one_frame_sign(img, is_in_mission)
            for i in range(7):
                if sign[0][i] == result:
                    sign[1][i] = sign[1][i] + 1

        for i in range(7):
            if sign[1][i] >= 3:
                sign2action = sign[0][i]
                break

        #sign2action이 뭐냐에 따라서 어떤 거 실행?

        if cv2.waitKey(1) & 0xff == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            break
