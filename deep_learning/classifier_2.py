import tensorflow as tf
import time
import sys

t1 = time.time()
#Evaluate 할 이미지 디렉토리
image_path = 'C:/Users/Jonkim/Desktop/slim/user_images/1.png'

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

#Smart car Label 삽입
label_lines = [line.rstrip() for line in tf.gfile.GFile("C:/Users/Jonkim/Desktop/tmp/smartcar/labels.txt")]

# freezing pb file 디렉토리 (이거도 push 했음 확인바람)
with tf.gfile.FastGFile("C:/Users/Jonkim/Desktop/slim/minimal_graph.proto", 'rb') as f:
    #with tf.device('/gpu:0'):
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

#config=tf.ConfigProto(log_device_placement=True)

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    # Feed data tensor 이름 각각 입력 (softmax_tensor가 y_eval, predictions가 x 데이터인듯)
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV1/Logits/Predictions/Softmax:0')
    predictions = sess.run(softmax_tensor, {'input_image:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    print(top_k)

    cnt = 0
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print()
        print('%s (score = %.5f)' % (human_string, score))
        cnt = cnt + 1
        # if(cnt == 3):
        #        break
print()
t2 = time.time()
print(t2 - t1)