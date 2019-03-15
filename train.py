import tensorflow as tf
import numpy as np
import net
import matplotlib.pyplot as plt

data = []
label = []

for i in range(1000):
    d = np.random.normal(0, 1, size=40)
    data.append(d)
    sum0 = []
    for j in range(5):
        s = 0
        ii = j * 8
        for k in range(8):
            s += d[ii + k]
        sum0.append(s)

    ll = 0
    last_sum = sum0[0]
    for j in range(1, 5):
        if last_sum <= sum0[j]:
            ll += 1
        else:
            ll = 0
        if ll >= 3:
            break
        last_sum = sum0[j]

    if ll >= 3:
        label.append([1, 0])
    else:
        label.append([0, 1])

data = np.array(data)
label = np.array(label)

print(data.shape)
print(label.shape)

X = tf.placeholder(dtype=tf.float64, shape=[None, 40], name='X')
Y = tf.placeholder(dtype=tf.float64, shape=[None, 2], name='Y')
S = tf.placeholder(dtype=tf.float64, shape=[None, net.STATE_LEN], name='S')

global_step = tf.Variable(0, trainable=False)

STARTER_LEARNING_RATE = 0.00005
DECAY_STEPS = 100
DECAY_RATE = 0.99
MOVING_AVERAGE_DECAY = 0.99


prediction = net.inference(X, S)

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
weights = graph.get_tensor_by_name('w:0')
# weights0_avg = graph.get_tensor_by_name('fully_connected0/weights/ExponentialMovingAverage:0')
# biases0_avg = variable_averages.average(biases0)

acc = []

with tf.control_dependencies([opt_op]):
    training_op = tf.group(variables_averages_op)
with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for epoch in range(3000):

        a = sess.run(accuracy, feed_dict={
            X: data[100:, :],
            Y: label[100:, :],
            S: np.zeros([label.shape[0]-100, net.STATE_LEN])})

        acc.append(a)
        if epoch % 100 == 0:
         print(str(epoch) + '  ' + str(a))

        sess.run(training_op, feed_dict={
            X: data[0:100, :],
            Y: label[0:100, :],
            S: np.zeros([100, net.STATE_LEN])})

    print(sess.run(weights))

acc = np.array(acc)
plt.plot([i for i in range(acc.shape[0])], acc)
plt.yticks([float(i) * 0.1 for i in range(11)])
plt.show()
