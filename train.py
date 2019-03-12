import tensorflow as tf
import numpy as np
import net

data = []
label = []

for i in range(1000):
    d = np.random.normal(0, 1, size=4)
    data.append(d)
    if d[0] + d[2] >= d[1] + d[3]:
        label.append([1])
    else:
        label.append([0])

data = np.array(data)
label = np.array(label)

print(data.shape)
print(label.shape)

X = tf.placeholder(dtype=tf.float64, shape=[None, 4], name='X')
Y = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='Y')

prediction = net.inference(X)
total_loss = tf.lo

