from matplotlib import pyplot as plt
import sys
import numpy as np
import os
import tensorflow as tf
import time
'''
우선 구현 방법은 Tensorflow/model의 slim이라는 tensorflow가 제공하는 틀을 이용할 거임
이 방법을 사용한 이유는 실제로 데이터를 구현하고 tensorflow로 표지판을 인식하는데까지 너무 많은 노력과 지식이 필요한데
그것을 충당할 수 있는 시간이 없어서 tensorflow에서 제공하는 모듈을 사용하기로 함
'''
t = time.time()
sys.path.insert(0, 'C:/Users/Administrator/Desktop/HEVEN_AutonomousCar_2018/slim')
#이 부분이 중요! 아래에 nets와 preprocessing은 tensorflow/model안에 slim이라는 폴더 안에 있는 폴더로써 slim파일을 불러와야 작동이 됨
#따라서 만약 Tensorflow/model파일이 없으면 https://github.com/tensorflow/models/ 여기에 들어가서 다운받은후에 slim 디렉토리를 위에 넣어줌

from nets import inception
from preprocessing import inception_preprocessing

checkpoints_dir = 'C:/Users/Administrator/Desktop/tmp/train_inception_v1_smartcar_logs'
#데이터의 checkpoint 디렉토리 넣어줌
slim = tf.contrib.slim

image_size = inception.inception_v1.default_image_size
#사용되는 딥러닝 툴은 inception v1으로 가동됨
with tf.Graph().as_default():


    images_dir = 'C:/Users/Administrator/Desktop/HEVEN_AutonomousCar_2018/slim/user_images'
    #자신이 원하는 이미지 인식, slim 폴더안에 user_images폴더를 만들고 그 안에 인식하고 싶은 사진을 넣어주면 사진 수 만큼 인식할거임
    user_images = []
    user_processed_images = []

    image_files = os.listdir(images_dir)
    #위에 입력한 디렉토리의 사진 읽어드림
    for i in image_files:
        image_input = tf.read_file(images_dir +"/"+ i)
        #디렉토리에 있는 사진 하나씩 처리하는 과정
        #어떠한 이미지 파일을 넣어도 Input에 맞게 사진을 반환해줌
        image = tf.image.decode_jpeg(image_input, channels=3)
        user_images.append(image)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        user_processed_images.append(processed_image)

    processed_images  = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(user_processed_images, num_classes=7, is_training=False)
        #Number of class: 우리 표지판 총 7개의 class를 판별
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'model.ckpt-11542'),
        #Checkpoint 디렉토리에서 실제로 사용되는 최신 데이터
        slim.get_model_variables('InceptionV1'))
        #Slim model중 InceptionV1을 이용함

    with tf.Session() as sess:
        init_fn(sess)
        np_images, probabilities = sess.run([user_images, probabilities])

    names = os.listdir("C:/Users/Administrator/Desktop/dataset/smartcar/smartcar_photos")
    #7개 class의 이름을 불러오는 작업, smartcar_photos안에 총 7개의 표지판 이름으로 된 폴더가 있는데 그 이름들을 인식함

    count = 0
    for files in range(len(image_files)):
        probabilitie = probabilities[files, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilitie), key=lambda x:x[1])]

        plt.figure()
        plt.imshow(np_images[files].astype(np.uint8))
        plt.axis('off')
        plt.show()

        for p in range(7):
            index = sorted_inds[p]
            print('Probability %0.2f%% => [%s]' % (probabilitie[index], names[index]))
            if(probabilitie[index] >= 0.95):
                count += 1
        print()
    print (count)
t2 = time.time()

print(t2 - t)