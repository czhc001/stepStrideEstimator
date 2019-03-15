import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

CHANNEL0 = 64
WIDTH0 = 6
HEIGHT0 = 256

FULLY_WIDTH0 = 2

FULLY_WIDTH1 = 2


def inference(inputs, init_state):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        # net = tf.reshape(net, [-1, ])
        # net = slim.conv2d(inputs, CHANNEL0, [WIDTH0, HEIGHT0])

        inputs_slice = []
        net_slice = []
        weights_rnn = tf.Variable(tf.random_normal([12, 4], stddev=1, dtype=tf.float64), name='w')
        biases_rnn = tf.Variable(tf.random_normal([1, 4], stddev=0, dtype=tf.float64, mean=0.1), name='b')

        state = init_state
        for i in range(5):
            ns = inputs[:, i * 8:i * 8 + 8]
            inputs_slice.append(ns)
            state = __cell(ns, state, weights_rnn, biases_rnn)
            net_slice.append(state)
        print(state.get_shape())
        net = tf.concat([net_slice[i] for i in range(5)], 1)
        net = slim.fully_connected(net, FULLY_WIDTH1, scope="fully_connected2", activation_fn=None)
        return net


def __cell(cell_input, state, weights, biases,):
    net = tf.tanh(tf.matmul(tf.concat([state, cell_input], 1), weights) + biases)
    return net
