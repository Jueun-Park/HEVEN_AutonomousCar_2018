# 코드에 대한 자세한 설명
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/pros/

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf


# <다중 계층 합성곱 신경망, Deep CNN(convolutional neural network)>

# 가중치 초기화
# 합성곱 신경망 모델을 구성하기 위해서는 많은 수의 가중치와 편향을 사용하게 됩니다.
# 매번 모델을 만들 때마다 반복하는 대신,
# 아래 코드와 같이 이러한 일을 해 주는 함수 두 개를 생성합니다.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 합성곱(Convolution)과 풀링(Pooling)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])  # 28*28인 이미지를 한 줄로 펼친 크기
W = tf.Variable(tf.zeros([784, 10]))  # 가중치 # 입력 이미지 벡터의 크기가 784, 출력 숫자 클래스가 10개
b = tf.Variable(tf.zeros([10]))  # 바이어스 # 10차원 벡터
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 0부터 9까지의 숫자 10개 클래스
'''첫 번째 합성곱(Convolution) 계층'''
# 합성곱 계층에서는 5x5의 윈도우(patch라고도 함) 크기를 가지는 32개의 필터를 사용하며,
# 따라서 구조(shape)가 [5, 5, 1, 32]인 가중치 텐서를 정의해야 합니다.
# 처음 두 개의 차원은 윈도우의 크기, 세 번째는 입력 채널의 수,
# 마지막은 출력 채널의 수(즉, 얼마나 많은 특징을 사용할 것인가)를 나타냅니다.
W_conv1 = weight_variable([5, 5, 1, 32])
# 또한, 각각의 출력 채널에 대한 편향을 정의해야 합니다.
b_conv1 = bias_variable([32])

# x를 4차원 텐서로 reshape
# 두 번째와 세 번째 차원은 이미지의 가로와 세로 길이,
# 그리고 마지막 차원은 컬러 채널의 수를 나타냅니다.
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 모델
h_pool1 = max_pool_2x2(h_conv1)  # 출력 값을 구하기 위해 맥스 풀링 적용

'''두 번째 합성곱(Convolution) 계층'''
# 심층 신경망 구성
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''완전 연결 계층 (Fully-Connected Layer)'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 드롭아웃(Dropout)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''최종 소프트맥스 계층'''
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# <모델의 훈련 및 평가>

# ADAM 최적화 알고리즘을 사용
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(session=sess,
                                         feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# 프린트할 때 에러
# Process finished with exit code -1073740791 (0xC0000409)
