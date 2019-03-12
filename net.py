import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

CHANNEL0 = 64
WIDTH0 = 6
HEIGHT0 = 256

FULLY_WIDTH0 = 128

FULLY_WIDTH1 = 1


def inference(inputs):
    net = slim.conv2d(inputs, CHANNEL0, [WIDTH0, HEIGHT0])
    net = slim.fully_connected(net, FULLY_WIDTH0, scope="fully_connected0", activation_fn=tf.nn.relu)
    net = slim.fully_connected(net, FULLY_WIDTH1, scope="fully_connected0", activation_fn=tf.nn.sigmoid)
    return net
