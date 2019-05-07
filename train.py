import tensorflow as tf
import numpy as np
import net
import load
import matplotlib.pyplot as plt
import threading
import time
import sys


frame_size = load.FRAME_SIZE
data, label = load.load_data_and_label('D:\\IdeaProjects\\IdentitySample\\data\\sample1557248511752.csv')
train_end = int(data.shape[0] * 0.5)

X = tf.placeholder(dtype=tf.float64, shape=[None, frame_size], name='X')
Y_ = tf.placeholder(dtype=tf.int64, shape=[None], name='Y_')
Y = tf.one_hot(indices=Y_, depth=load.CLASS_NUM, on_value=1, off_value=0, name='Y')
print(Y.get_shape())
S = tf.placeholder(dtype=tf.float64, shape=[None, net.STATE_LEN], name='S')
P = tf.placeholder(dtype=tf.float64, name='P')

global_step = tf.Variable(0, trainable=False)

STARTER_LEARNING_RATE = 0.001
DECAY_STEPS = 100
DECAY_RATE = 0.99
MOVING_AVERAGE_DECAY = 0.99


prediction = net.inference(X, S, P)
print(prediction.name)

learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE, global_step, DECAY_STEPS, DECAY_RATE, staircase=False)

total_loss = tf.losses.softmax_cross_entropy(Y, prediction)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
opt_op = optimizer.minimize(total_loss)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
variables_averages_op = variable_averages.apply(tf.trainable_variables())

print(tf.trainable_variables())
graph = tf.get_default_graph()
for var in tf.global_variables():
    print(var.name)
graph = tf.get_default_graph()
# weights = graph.get_tensor_by_name('w:0')
# weights0_avg = graph.get_tensor_by_name('fully_connected0/weights/ExponentialMovingAverage:0')
# biases0_avg = variable_averages.average(biases0)

accuracy_train = []
accuracy_test = []

train_pause = False


def train_break():
    global train_pause
    input("Press any key to pause")
    train_pause = True


t = threading.Thread(target=train_break)
t.setDaemon(True)
t.start()

with tf.control_dependencies([opt_op]):
    training_op = tf.group(variables_averages_op)
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(30000):
        if train_pause:
            break
        atest = sess.run(accuracy, feed_dict={
            X: data[train_end:, :],
            Y_: label[train_end:],
            S: np.zeros([label.shape[0]-train_end, net.STATE_LEN]),
            P: 1})
        accuracy_train.append(atest)

        atrain = sess.run(accuracy, feed_dict={
            X: data[0:train_end, :],
            Y_: label[0:train_end],
            S: np.zeros([train_end, net.STATE_LEN]),
            P: 1})
        accuracy_test.append(atrain)
        if epoch % 100 == 0:
            print(str(epoch) + '    Training Set Accuracy: ' + str(atrain))
            print(str(epoch) + '    Test Set Accuracy:     ' + str(atest))

        sess.run(training_op, feed_dict={
            X: data[0:train_end, :],
            Y_: label[0:train_end],
            S: np.zeros([train_end, net.STATE_LEN]),
            P: 0.5})

    # print(sess.run(weights))

accuracy_train = np.array(accuracy_train)
accuracy_test = np.array(accuracy_test)
plt.plot([i for i in range(accuracy_train.shape[0])], accuracy_train, color='r')
plt.plot([i for i in range(accuracy_test.shape[0])], accuracy_test, color='g')
plt.yticks([float(i) * 0.1 for i in range(11)])
plt.legend()
plt.show()
