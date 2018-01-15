# 코드에 대한 자세한 설명
# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/beginners/

# MNIST data set 불러오기
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 원-핫 벡터는 단 하나의 차원에서만 1이고, 나머지 차원에서는 0인 벡터입니다.

'''회귀 구현하기'''
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])  # 28*28인 이미지를 한 줄로 펼친 크기
# 이는 'placeholder'로, 우리가 텐서플로우에서 연산을 실행할 때 값을 입력할 자리입니다
# 여기서는 784차원의 벡터로 변형된 MNIST 이미지의 데이터를 넣으려고 합니다.
# (여기서 None은 해당 차원의 길이가 어떤 길이든지 될 수 있음을 의미합니다)


W = tf.Variable(tf.zeros([784, 10]))  # 가중치 # 입력 이미지 벡터의 크기가 784, 출력 숫자 클래스가 10개
b = tf.Variable(tf.zeros([10]))  # 바이어스 # 10차원 벡터
# Variable: 서로 상호작용하는 연산으로 이루어진 텐서플로우 그래프 안에 존재하는,
# 수정 가능한 텐서
# 모두 0으로 이루어진 텐서로 초기화.
# W에 784차원의 이미지 벡터를 곱해서 각 클래스에 대한 증거값을 나타내는 10차원 벡터를 얻고자 함
# b는 그 10차원 벡터에 더하기 위해 [10]의 형태를 갖는 것

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 모델, Wx+b

'''학습'''
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 0부터 9까지의 숫자 10개 클래스
# 모델이 안 좋다는 것을 표현
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(x, W) + b, labels=y_))
# 모델이 안 좋다는 것을 줄이는 방향으로
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):  # 1000번 학습
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 반복되는 루프의 각 단계마다, 우리는 학습 데이터셋에서 무작위로 선택된
    # 100개의 데이터로 구성된 "배치(batch)"를 가져옵니다.
    # 그 다음엔 placeholder의 자리에 데이터를 넣을 수 있도록
    # train_step을 실행하여 배치 데이터를 넘깁니다.
    # 무작위 데이터의 작은 배치를 사용하는 방법을
    # 확률적 학습(stochastic training)이라고 부릅니다
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''모델 평가하기'''
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 이 방식은 별로 정확도가 높지 않습니다. (약 92%)
print("accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))