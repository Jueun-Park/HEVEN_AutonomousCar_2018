from matplotlib import pyplot as plt
import sys
import numpy as np
import os
import tensorflow as tf

sys.path.insert(0, 'C:/Users/Administrator/Desktop/slim')

from nets import inception
from preprocessing import inception_preprocessing

checkpoints_dir = 'C:/Users/Administrator/Desktop/dataset/train_inception_v1_smartcar_FineTune_logs/all'

slim = tf.contrib.slim

image_size = inception.inception_v1.default_image_size    

with tf.Graph().as_default():

    images_dir = 'C:/Users/Administrator/Desktop/slim/user_images'
    user_images = [] 
    user_processed_images = [] 

    image_files = os.listdir(images_dir)

    for i in image_files:
        image_input = tf.read_file(images_dir +"/"+ i)
        image = tf.image.decode_jpeg(image_input, channels=3)
        user_images.append(image)
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        user_processed_images.append(processed_image)
        
    processed_images  = tf.expand_dims(processed_image, 0)
    
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(user_processed_images, num_classes=7, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'model.ckpt-500'),
        slim.get_model_variables('InceptionV1'))

    with tf.Session() as sess:
        init_fn(sess)
        np_images, probabilities = sess.run([user_images, probabilities])
    
    names = os.listdir("C:/Users/Administrator/Desktop/dataset/smartcar/smartcar_photos")
    

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
        print()