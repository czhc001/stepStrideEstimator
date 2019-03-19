import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

CHANNEL0 = 64
WIDTH0 = 6
HEIGHT0 = 256

SLICE_LEN = 8
SLICE_COUNT = int(40/SLICE_LEN)
STATE_LEN = 5

FULLY_WIDTH0 = 2

FULLY_WIDTH1 = 2


def inference(inputs, init_state):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        # net = tf.reshape(net, [-1, ])
        # net = slim.conv2d(inputs, CHANNEL0, [WIDTH0, HEIGHT0])
        net = inputs

        net = slim.fully_connected(net, FULLY_WIDTH1, scope="fully_connected2", activation_fn=None)
        return net


def __cell(cell_input, state, weights, biases,):
    net = tf.nn.relu(tf.matmul(tf.concat([state, cell_input], 1), weights) + biases)
    return net
