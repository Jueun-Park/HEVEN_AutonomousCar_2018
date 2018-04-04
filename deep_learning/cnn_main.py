import tensorflow as tf
import numpy as np
import re

input_h = 224
input_w = 224
input_ch = 3
n_output = 8
logs_path = "/Image_training_files"

def get_input_queue(csv_filename, num_epochs=None):

    train_images = []
    train_labels = []

    for line in open(csv_filename,'r'):
        cols = re.split(',|\n',line)
        train_images.append(cols[0])
        train_labels.append(int(cols[2]))  # 3rd column is label and need to be converted to int type

    input_queue = tf.train.slice_input_producer([train_images,train_labels],num_epochs = num_epochs, shuffle= True)

    return input_queue

def read_data(input_queue):
    image_file = input_queue[0]
    label = input_queue[1]

    image = tf.image.decode_jpeg(tf.read_file(image_file),channels=3)

    return image,label,image_file

def read_data_batch(csv_filename,batch_size):
    input_queue = get_input_queue(csv_filename)
    image,label,filename = read_data(input_queue)
    image = tf.reshape(image,[224, 224, 3])

    batch_image, batch_label, batch_file = tf.train.batch([image,label,filename], batch_size= batch_size)

    batch_file = tf.reshape(batch_file,[batch_size,1])
    batch_label_on_hot = tf.one_hot(tf.to_int64(batch_label), n_output, on_value=1.0, off_value=0.0)

    return batch_image,batch_label_on_hot,batch_file

with tf.name_scope('weights'):
    weights = {
        'conv1' : tf.Variable(tf.random_normal([11,11,3,96], stddev=0.1)),
        'conv2' : tf.Variable(tf.random_normal([5,5,96,256], stddev=0.1)),
        'conv3' : tf.Variable(tf.random_normal([3,3,256,384], stddev=0.1)),
        'conv4' : tf.Variable(tf.random_normal([3,3,384,192], stddev=0.1)),
        'conv5' : tf.Variable(tf.random_normal([3,3,192,256], stddev=0.1)),
        'fc1' : tf.Variable(tf.random_normal([9216,4096], stddev=0.1)),
        'fc2' : tf.Variable(tf.random_normal([4096,1000], stddev=0.1)),
        'output' : tf.Variable(tf.random_normal([1000,n_output],stddev=0.1))
    }

with tf.name_scope('biases'):
    biases = {
        'conv1' : tf.Variable(tf.random_normal([96], stddev=0.1)),
        'conv2' : tf.Variable(tf.random_normal([256], stddev=0.1)),
        'conv3' : tf.Variable(tf.random_normal([384], stddev=0.1)),
        'conv4' : tf.Variable(tf.random_normal([192], stddev=0.1)),
        'conv5' : tf.Variable(tf.random_normal([256], stddev=0.1)),
        'fc1' : tf.Variable(tf.random_normal([4096], stddev=0.1)),
        'fc2' : tf.Variable(tf.random_normal([1000], stddev=0.1)),
        'output' : tf.Variable(tf.random_normal([n_output],stddev=0.1))
    }

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, input_h, input_w, input_ch])
    y = tf.placeholder(tf.float32, [None, n_output])
    y_ = tf.placeholder(tf.float32, [None, n_output])

def net(x, weights, biases):
    with tf.name_scope('conv1'):
        conv1 = tf.nn.conv2d(x, weights['conv1'], strides=[1,4,4,1],padding='SAME')
        conv1 = tf.nn.relu(tf.add(conv1,biases['conv1']))
        conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5,bias=2,alpha=10^-4,beta=0.75)
        conv1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')

    with tf.name_scope('conv2'):
        conv2 = tf.nn.conv2d(conv1, weights['conv2'], strides=[1,1,1,1], padding='VALID')
        conv2 = tf.nn.relu(tf.add(conv2,biases['conv2']))
        conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=2, alpha=10 ^ -4, beta=0.75)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv3'):
        conv3 = tf.nn.conv2d(conv2, weights['conv3'], strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.relu(tf.add(conv3,biases['conv3']))

    with tf.name_scope('conv4'):
        conv4 = tf.nn.conv2d(conv3, weights['conv4'], strides=[1,1,1,1], padding='SAME')
        conv4 = tf.nn.relu(tf.add(conv4,biases['conv4']))

    with tf.name_scope('conv5'):
        conv5 = tf.nn.conv2d(conv4, weights['conv5'], strides=[1,1,1,1], padding='SAME')
        conv5 = tf.nn.relu(tf.add(conv2,biases['conv5']))
        conv5 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv5 = tf.reshape(conv5, [-1,9216])

    with tf.name_scope('fc1'):
        fc1 = tf.add(tf.matmul(conv5, weights['fc1'],biases['fc1']))
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,keep_prob=0.75)

    with tf.name_scope('fc2'):
        fc2 = tf.add(tf.matmul(fc1, weights['fc2'],biases['fc2']))
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2,keep_prob=0.75)

        output = tf.add(tf.matmul(fc2,weights['output'], biases['output']))

        return output

pred = net(x,weights,biases)

with tf.name_scope('loss'):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)
    loss = tf.reduce_mean(loss)

tf.summary.scalar("loss",loss)

global_step = tf.Variable(0,trainable=False)
starter_lr = 0.01
lr = tf.train.exponential_decay(starter_lr,global_step,10000,0.96,staircase=True)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step)

n_batch = 64
n_iter = 1000000
n_prt = 240


