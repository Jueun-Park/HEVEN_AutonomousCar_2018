'''
1. 아직 데이터셋이 없어서 검증 불가
2. input image의 크기에 따라서 모델 수정 가능성
3. 데이터셋의 크기에 따라 accuracy가 낮아질 수 있어서 그때에도 hyperparameter (ex.learning rate, batch size ... ) 및 모델 수정 가능성
4. 한 파일에서 데이터를 불러오고, 전처리를 하고, 모델을 만들고, 학습을 진행하기 때문에 분산해서 만들 생각? 도 있음
'''

import tensorflow as tf
# Load pickled data
import pickle
from matplotlib import pyplot
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
#from skimage import exposure

#현재 데이터셋이 없어서 검증 불가
'''
training_file = "/Users/harshit.sharma/Desktop/udacity_selfdriving/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p"
validation_file = "/Users/harshit.sharma/Desktop/udacity_selfdriving/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/valid.p"
testing_file = "/Users/harshit.sharma/Desktop/udacity_selfdriving/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
'''

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
# TODO: 모든 데이터셋의 shape가 동일해야함.
image_shape = np.array(X_train[0]).shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)

pyplot.bar( np.arange( 43 ), class_counts, align='center' )
pyplot.xlabel('Class')
pyplot.ylabel('Number of training examples')
pyplot.xlim([-1, 43])
pyplot.show()
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("Number of validation examples=",n_validation)



index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])



# Data preprocessing : 이미지 전처리
def preprocess_dataset(X, y = None):
    #print(X.shape)
    #Convert to grayscale, e.g. single Y channel
    X = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]
    #Scale features to be in [0, 1]
    X = (X / 255.).astype(np.float32)
    # Add a single grayscale channel
    X = X.reshape(X.shape + (1,))
    return X, y
X_test, y_test = preprocess_dataset(X_test,y_test)
X,y = X_train,y_train
X_train,y_train = preprocess_dataset(X_train,y_train)
#print(X_train.shape)
X_train, y_train = shuffle(X_train, y_train)
#X_train = (X_train -128)/128
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20)


EPOCHS = 40
BATCH_SIZE = 128
dropout1 = 0.90 # Dropout, probability to keep units
dropout2 = 0.80
dropout3 = 0.70
dropout4 = 0.50

from tensorflow.contrib.layers import flatten

mu = 0
sigma = 0.1
layer1_weight = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=mu, stddev=sigma))
layer1_bias = tf.Variable(tf.zeros(6))
layer2_weight = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma))
layer2_bias = tf.Variable(tf.zeros(16))
flat_weight = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
bias_flat = tf.Variable(tf.zeros(120))
flat_weight2 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
bias_flat2 = tf.Variable(tf.zeros(84))
flat_weight3 = tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma))
bias_flat3 = tf.Variable(tf.zeros(43))


def LeNet(x, train=True):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    x = tf.nn.conv2d(x, layer1_weight, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(x, layer1_bias)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if (train):
        x = tf.nn.dropout(x, dropout1)

    x = tf.nn.conv2d(x, layer2_weight, strides=[1, 1, 1, 1], padding='VALID')
    x = tf.nn.bias_add(x, layer2_bias)
    x = tf.nn.relu(x)
    conv2 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if (train):
        conv2 = tf.nn.dropout(conv2, dropout2)

    fc0 = flatten(conv2)
    fc1 = tf.add(tf.matmul(fc0, flat_weight), bias_flat)
    fc1 = tf.nn.relu(fc1)
    if (train):
        fc1 = tf.nn.dropout(fc1, dropout3)

    fc1 = tf.add(tf.matmul(fc1, flat_weight2), bias_flat2)
    fc1 = tf.nn.relu(fc1)
    if (train):
        fc1 = tf.nn.dropout(fc1, dropout4)
    fc1 = tf.add(tf.matmul(fc1, flat_weight3), bias_flat3)
    logits = tf.nn.relu(fc1)

    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
one_hot_y = tf.one_hot(y, 43)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)

training_operation = optimizer.minimize(loss_operation)
logits_2 = LeNet(x)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, './lenet')
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))