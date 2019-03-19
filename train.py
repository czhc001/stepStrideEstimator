import tensorflow as tf
import numpy as np
import net
import load
import matplotlib.pyplot as plt

data, label = load.load_data_and_label()

X = tf.placeholder(dtype=tf.float64, shape=[None, 40], name='X')
Y = tf.placeholder(dtype=tf.float64, shape=[None, 3], name='Y')
S = tf.placeholder(dtype=tf.float64, shape=[None, net.STATE_LEN], name='S')
P = tf.placeholder(dtype=tf.float64, name='P')

global_step = tf.Variable(0, trainable=False)

STARTER_LEARNING_RATE = 5.0
DECAY_STEPS = 100
DECAY_RATE = 0.99
MOVING_AVERAGE_DECAY = 0.99


prediction = net.inference(X, S, P)

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

acc = []
acct = []

with tf.control_dependencies([opt_op]):
    training_op = tf.group(variables_averages_op)
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(30000):

        a = sess.run(accuracy, feed_dict={
            X: data[1000:, :],
            Y: label[1000:, :],
            S: np.zeros([label.shape[0]-1000, net.STATE_LEN]),
            P: 1})
        acc.append(a)

        at = sess.run(accuracy, feed_dict={
            X: data[0:1000, :],
            Y: label[0:1000, :],
            S: np.zeros([1000, net.STATE_LEN]),
            P: 1})
        acct.append(at)
        if epoch % 100 == 0:
            print(str(epoch) + '  ' + str(a) + '  ' + str(at))

        sess.run(training_op, feed_dict={
            X: data[0:1000, :],
            Y: label[0:1000, :],
            S: np.zeros([1000, net.STATE_LEN]),
            P: 0.5})

    # print(sess.run(weights))

acc = np.array(acc)
acct = np.array(acct)
plt.plot([i for i in range(acc.shape[0])], acc, color='r')
plt.plot([i for i in range(acct.shape[0])], acct, color='g')
plt.yticks([float(i) * 0.1 for i in range(11)])
plt.show()
