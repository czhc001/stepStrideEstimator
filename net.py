import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import load

# INPUT_LEN = load.FRAME_SIZE

CHANNEL0 = load.FRAME_HEIGHT
WIDTH0 = load.FRAME_WIDTH
HEIGHT0 = 1

CONV0_W = 20
CONV0_H = 1
CONV0_C = 16

SLICE_LEN = 8
SLICE_COUNT = int(40/SLICE_LEN)
STATE_LEN = 5

FULLY_WIDTH0 = 250

OUTPUT_SIZE = load.CLASS_NUM


def inference(inputs, init_state, keep_prob):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(10.0)):
        net = inputs
        net = tf.reshape(inputs, [-1, HEIGHT0, WIDTH0, CHANNEL0])
        net = slim.conv2d(net, CONV0_C, [CONV0_W, CONV0_H])
        net = tf.reshape(net, [-1, net.get_shape()[1] * net.get_shape()[2] * net.get_shape()[3]])
        net = tf.nn.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net, FULLY_WIDTH0, scope="fully_connected1", activation_fn=tf.nn.relu)
        net = tf.nn.dropout(net, keep_prob=keep_prob)
        net = slim.fully_connected(net, OUTPUT_SIZE, scope="output", activation_fn=None)
        return net


def __cell(cell_input, state, weights, biases,):
    net = tf.nn.relu(tf.matmul(tf.concat([state, cell_input], 1), weights) + biases)
    return net
