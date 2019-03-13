import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

CHANNEL0 = 64
WIDTH0 = 6
HEIGHT0 = 256

FULLY_WIDTH0 = 16

FULLY_WIDTH1 = 2


def inference(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        # net = tf.reshape(net, [-1, ])
        # net = slim.conv2d(inputs, CHANNEL0, [WIDTH0, HEIGHT0])
        net = slim.fully_connected(inputs, FULLY_WIDTH0, scope="fully_connected0", activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, FULLY_WIDTH1, scope="fully_connected1", activation_fn=None)
        return net
