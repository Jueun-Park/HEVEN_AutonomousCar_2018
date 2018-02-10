# CNN Train & Save
# http://hugrypiggykim.com/2016/09/04/tensorflow-%EA%B8%B0%EB%B3%B8%EB%AC%B8%EB%B2%95-5-cnn-train-save/

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)


# 원하는 행렬 사이즈로 초기 값을 만들어서 리턴하는 메서드
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 0.1 로 초기값 지정하여 원하는 사이즈로 리턴
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides= [1, stride, stride, 1] 차원 축소 작업 시 마스크 매트릭스를 이동하는
    # padding='SAME' 다음 레벨에서도 매트릭스가 줄어들지 않도록 패딩을 추가한다
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])  # [filter_height, filter_width, in_channels, out_channels]
b_conv1 = bias_variable([32])


# x: [batch, height, width, channels]
# 2x2 행렬에 가장 큰 값을 찾아서 추출, 가로세로 2칸씩 이동.
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Layer1: CNN + MaxPool

# 인풋 데이터 매트릭스를 변형한다.
# 784개의 행렬을 갖는 복수의 데이터를 [-1, 28, 28, 1] 의 형태로 변형한다.
# 테스트 데이터 수 만큼
# (-1), [28x28] 행렬로 만드는데 각 픽셀 데이터는 RGB 데이터가 아니고 하나의 값만 갖도록 변환
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Layer2

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Layer3

# 현재 최종 데이터의 수는 7 * 7 * 64 = 3136개이지만 1024개를 사용한다.
# (1024는 임의의 선택 값이다)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Drop Out
keep_prob = tf.placeholder(tf.float32)  # drop out 연산의 결과를 담을 변수
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Output Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 최적화

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
Train & Save Model
"""
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

# 50개씩 20000번 반복 학습
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 10회 단위로 한 번씩 모델 정합성 테스트
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    # batch[0] 28x28 이미지, batch[1] 숫자 태그, keep_prob: Dropout 비율
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# save_path = saver.save(sess, "model.ckpt")
# print("Model saved in file: ", save_path)
# tensorflow.python.framework.errors_impl.NotFoundError: Failed to create a directory: ; No such file or directory
# tensorflow.python.framework.errors_impl.NotFoundError: Failed to create a directory: ; No such file or directory
# NotFoundError (see above for traceback): Failed to create a directory: ; No such file or directory
# ValueError: Parent directory of model.ckpt doesn't exist, can't save.

# 최종적으로 모델의 정합성을 체크한다
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close()
