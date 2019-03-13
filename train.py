import tensorflow as tf
import numpy as np
import net
import matplotlib.pyplot as plt

data = []
label = []

for i in range(1000):
    d = np.random.normal(0, 1, size=4)
    data.append(d)
    if d[0] + d[2] >= d[1] + d[3]:
        label.append([1, 0])
    else:
        label.append([0, 1])

data = np.array(data)
label = np.array(label)

print(data.shape)
print(label.shape)

X = tf.placeholder(dtype=tf.float64, shape=[None, 4], name='X')
Y = tf.placeholder(dtype=tf.float64, shape=[None, 2], name='Y')

global_step = tf.Variable(0, trainable=False)

STARTER_LEARNING_RATE = 0.1
DECAY_STEPS = 10
DECAY_RATE = 0.96
MOVING_AVERAGE_DECAY = 0.99


prediction = net.inference(X)

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
weights0 = graph.get_tensor_by_name('fully_connected0/weights:0')
biases0 = graph.get_tensor_by_name('fully_connected0/biases:0')
weights0_avg = variable_averages.average(weights0)
biases0_avg = variable_averages.average(biases0)

acc = []

with tf.control_dependencies([opt_op]):
    training_op = tf.group(variables_averages_op)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(300):
        a = sess.run(accuracy, feed_dict={X: data[700:-1, :], Y: label[700:-1, :]})
        acc.append(a)
        print(str(epoch) + '  ' + str(a))
        sess.run(training_op, feed_dict={X: data[0:700, :], Y: label[0:700, :]})
    w0 = sess.run(weights0)
    w0_ = sess.run(weights0_avg)
    print(w0)
    print(w0_)

acc = np.array(acc)
plt.plot([i for i in range(acc.shape[0])], acc)
plt.yticks([float(i) * 0.1 for i in range(11)])
plt.show()
